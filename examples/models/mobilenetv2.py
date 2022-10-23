# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchquanter.nn import QConvBNReLU, QAdd, QReLU, QMean, QLinear, QAdaptiveAvgPool2d


class BaseBlock(nn.Module):
    alpha = 1
    def __init__(self, input_channel, output_channel, t = 6, downsample = False, **kwargs):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 
        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        # for main path:
        c = t * input_channel
        # 1x1   point wise conv
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        # main path
        x = F.relu(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))
        # shortcut path
        x = x + inputs if self.shortcut else x
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.qconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.qconv3 = QConvBNReLU(self.conv3, self.bn3, relu=False, qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.qadd = QAdd(qi1=False, qi2=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, inputs):
        x = inputs
        x = self.qconv1(x)
        x = self.qconv2(x)
        x = self.qconv3(x)
        if self.shortcut:
            x = self.qadd(x, inputs)
        return x

    def freeze(self, input_qi):
        self.qconv1.freeze(input_qi)
        self.qconv2.freeze(self.qconv1.qo)
        self.qconv3.freeze(self.qconv2.qo)
        if self.shortcut:
            self.qadd.freeze(self.qconv3.qo, input_qi)
            return self.qadd.qo
        return self.qconv3.qo

    def quantize_inference(self, qx_in, mode=None):
        qx = qx_in
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx = self.qconv2.quantize_inference(qx, mode=mode)
        qx = self.qconv3.quantize_inference(qx, mode=mode)
        if self.shortcut:
            qx = self.qadd.quantize_inference(qx, qx_in)
        return qx 


class MobileNetV2(nn.Module):
    def __init__(self, output_size = 10, alpha = 1, **kwargs):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):
        # first conv layer
        x = F.relu(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])
        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])
        # last conv layer
        x = F.relu(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])
        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True, symmetric_feature=False):
        self.qconv0 = QConvBNReLU(self.conv0, self.bn0, qi=True, qo=True,
            num_bits=num_bits, signed=signed, symmetric_feature=symmetric_feature)
        for i in range(len(self.bottlenecks)):
            self.bottlenecks[i].quantize(symmetric_feature=symmetric_feature)
        self.qconv1 = QConvBNReLU(self.conv1, self.bn1, qi=False, qo=True,
            num_bits=num_bits, signed=signed, symmetric_feature=symmetric_feature)
        self.qavg = QAdaptiveAvgPool2d(1, qi=False, qo=True,
            num_bits=num_bits, signed=signed, symmetric_feature=symmetric_feature)
        # self.qavg = QMean(dim=[-1, -2], keepdim=True, qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, relu=False,
            num_bits=num_bits, signed=signed, symmetric_feature=symmetric_feature)

    def quantize_forward(self, x):
        x = self.qconv0(x)
        for i in range(len(self.bottlenecks)):
            x = self.bottlenecks[i].quantize_forward(x)
        x = self.qconv1(x)
        x = self.qavg(x)
        x = x.view(x.shape[0], -1)
        x = self.qfc(x)
        return x

    def freeze(self):
        """
        统计完min、max后将网络彻底变成int8，例如将weight、bias变成int8
        """
        self.qconv0.freeze()
        tmp_qo = self.qconv0.qo
        for i in range(len(self.bottlenecks)):
            tmp_qo = self.bottlenecks[i].freeze(tmp_qo)
        self.qconv1.freeze(tmp_qo)
        self.qavg.freeze(self.qconv1.qo)
        # self.qfc.freeze(self.qconv1.qo)
        self.qfc.freeze(self.qavg.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        """
        真正的量化推理，使用int8
        """
        qx = self.qconv0.qi.quantize_tensor(x)
        qx = self.qconv0.quantize_inference(qx, mode=mode)
        
        for i in range(len(self.bottlenecks)):
            qx = self.bottlenecks[i].quantize_inference(qx, mode=mode)
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx = self.qavg.quantize_inference(qx, mode=mode)
        # out = self.qavg.qo.dequantize_tensor(qx)
        qx = qx.view(qx.shape[0], -1)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


def mobilenetv2_quant(pretrained=False, **kwargs):
    num_classes = kwargs['num_classes']
    return MobileNetV2(num_classes)
