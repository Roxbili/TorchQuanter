"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import functional as F
from torchquanter.nn import QConvBNReLU, QAdd, QReLU, QMean, QLinear


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=True), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) +
                                     self.shortcut(x))

    def quantize(self, first_qi=True, num_bits=8, signed=True, symmetric_feature=False):
        self.qresidual_function = nn.Sequential(
            QConvBNReLU(
                self.residual_function[0],
                self.residual_function[1],
                qi=first_qi, qo=True,
                num_bits=num_bits, signed=signed,
                symmetric_feature=symmetric_feature
            ),
            QConvBNReLU(
                self.residual_function[3],
                self.residual_function[4],
                relu=False, qi=False, qo=True,
                num_bits=num_bits, signed=signed,
                symmetric_feature=symmetric_feature
            )
        )
        if len(self.shortcut) > 0:
            self.qshortcut = QConvBNReLU(
                self.shortcut[0], self.shortcut[1], relu=False,
                qi=False, qo=True,
                num_bits=num_bits, signed=signed,
                symmetric_feature=symmetric_feature
            )
        self.qadd = QAdd(qi1=False, qi2=False, qo=True, num_bits=num_bits,
            signed=signed, symmetric_feature=symmetric_feature)
        self.qrelu = QReLU(qi=False, num_bits=num_bits, signed=signed, symmetric_feature=symmetric_feature)

    def quantize_forward(self, x):
        x1 = self.qresidual_function(x)
        if len(self.shortcut) > 0:
            x2 = self.qshortcut(x)
        else:
            x2 = x
        x = self.qadd(x1, x2)
        x = self.qrelu(x)
        return x

    def freeze(self, qi=None):
        self.qresidual_function[0].freeze(qi=qi)
        self.qresidual_function[1].freeze(qi=self.qresidual_function[0].qo)
        if len(self.shortcut) > 0:
            self.qshortcut.freeze(qi=qi)
            self.qadd.freeze(qi1=self.qresidual_function[1].qo, qi2=self.qshortcut.qo)
        else:
            self.qadd.freeze(qi1=self.qresidual_function[1].qo, qi2=qi)
        self.qrelu.freeze(self.qadd.qo)

    def quantize_inference(self, qx, mode='cmsis_nn'):
        qx1 = self.qresidual_function[0].quantize_inference(qx, mode=mode)
        qx1 = self.qresidual_function[1].quantize_inference(qx1, mode=mode)
        if len(self.shortcut) > 0:
            qx2 = self.qshortcut.quantize_inference(qx, mode=mode)
        else:
            qx2 = qx
        qx = self.qadd.quantize_inference(qx1, qx2, mode=mode)
        qx = self.qrelu.quantize_inference(qx)
        return qx


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) +
                                     self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def quantize(self, num_bits=8, signed=True, symmetric_feature=False):
        self.qconv1 = QConvBNReLU(self.conv1[0], self.conv1[1], qi=True,
                                  num_bits=num_bits, signed=signed,
                                  symmetric_feature=symmetric_feature)
        # conv2_x
        for block in self.conv2_x:
            block.quantize(first_qi=False, num_bits=num_bits, signed=signed,
                           symmetric_feature=symmetric_feature)
        # conv3_x
        for block in self.conv3_x:
            block.quantize(first_qi=False, num_bits=num_bits, signed=signed,
                           symmetric_feature=symmetric_feature)
        # conv4_x
        for block in self.conv4_x:
            block.quantize(first_qi=False, num_bits=num_bits, signed=signed,
                           symmetric_feature=symmetric_feature)
        # conv5_x
        for block in self.conv5_x:
            block.quantize(first_qi=False, num_bits=num_bits, signed=signed,
                           symmetric_feature=symmetric_feature)
        self.qavg_pool = QMean(dim=[-1, -2], keepdim=True, qi=False, qo=True,
                               num_bits=num_bits, signed=signed,
                               symmetric_feature=symmetric_feature)
        self.qfc = QLinear(self.fc, relu=False, qi=False, qo=True,
                           num_bits=num_bits, signed=signed,
                           symmetric_feature=symmetric_feature)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        # conv2_x
        for block in self.conv2_x:
            x = block.quantize_forward(x)
        # conv3_x
        for block in self.conv3_x:
            x = block.quantize_forward(x)
        # conv4_x
        for block in self.conv4_x:
            x = block.quantize_forward(x)
        # conv5_x
        for block in self.conv5_x:
            x = block.quantize_forward(x)
        x = self.qavg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        # conv2_x
        last_qo = self.qconv1.qo
        for block in self.conv2_x:
            block.freeze(qi=last_qo)
            last_qo = block.qadd.qo
        # conv3_x
        for block in self.conv3_x:
            block.freeze(qi=last_qo)
            last_qo = block.qadd.qo
        # conv4_x
        for block in self.conv4_x:
            block.freeze(qi=last_qo)
            last_qo = block.qadd.qo
        # conv5_x
        for block in self.conv5_x:
            block.freeze(qi=last_qo)
            last_qo = block.qadd.qo
        self.qavg_pool.freeze(qi=last_qo)
        self.qfc.freeze(qi=self.qavg_pool.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        # conv2_x
        for block in self.conv2_x:
            qx = block.quantize_inference(qx, mode=mode)
        # conv3_x
        for block in self.conv3_x:
            qx = block.quantize_inference(qx, mode=mode)
        # conv4_x
        for block in self.conv4_x:
            qx = block.quantize_inference(qx, mode=mode)
        # conv5_x
        for block in self.conv5_x:
            qx = block.quantize_inference(qx, mode=mode)
        qx = self.qavg_pool.quantize_inference(qx, mode=mode)
        qx = qx.view(qx.size(0), -1)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        x = self.qfc.qo.dequantize_tensor(qx)
        return x


def resnet18_quant(pretrained=False, **kwargs):
    """ return a ResNet 18 object
    """
    num_classes = 10
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34_quant(pretrained=False, **kwargs):
    """ return a ResNet 34 object
    """
    num_classes = 10
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50_quant(pretrained=False, **kwargs):
    """ return a ResNet 50 object
    """
    num_classes = 10
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)


def resnet101_quant(pretrained=False, **kwargs):
    """ return a ResNet 101 object
    """
    num_classes = 10
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)


def resnet152_quant(pretrained=False, **kwargs):
    """ return a ResNet 152 object
    """
    num_classes = 10
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)
