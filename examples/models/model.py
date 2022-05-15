import torch
import torch.nn as nn
import torch.nn.functional as F

from .tinyformer import (
    Attention,
    MV2Block
)
from torchquanter.nn import (
    QConv2d, 
    QMaxPool2d, 
    QReLU, 
    QLinear, 
    QConvBNReLU, 
    QLinearReLU, 
    QAdd, 
    QLayerNorm, 
    QSoftmax, 
    QMul, 
    QMatmul,
    QLayerNormTFLite
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv1 = QConv2d(self.block[0], qi=True, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qrelu1 = QReLU(qi=False, signed=signed)
        self.qmaxpool1 = QMaxPool2d(self.block[2], qi=False, signed=signed)
        self.qconv2 = QConv2d(self.block[3], qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qrelu2 = QReLU(qi=False, signed=signed)
        self.qmaxpool2 = QMaxPool2d(self.block[5], qi=False, signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        """
        统计min、max，同时模拟量化误差
        """
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.qfc(x)
        return x

    def freeze(self):
        """
        统计完min、max后将网络彻底变成int8，例如将weight、bias变成int8
        """
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool1.freeze(self.qconv1.qo)
        self.qconv2.freeze(self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2.freeze(self.qconv2.qo)
        self.qfc.freeze(self.qconv2.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        """
        真正的量化推理，使用int8
        """
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx, mode=mode)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2.quantize_inference(qx)
        qx = qx.view(-1, 5 * 5 * 64)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class ModelBN(nn.Module):
    def __init__(self):
        super(ModelBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv1 = QConvBNReLU(self.block[0], self.block[1], qi=True, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qmaxpool1 = QMaxPool2d(self.block[3], signed=signed)
        self.qconv2 = QConvBNReLU(self.block[4], self.block[5], qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qmaxpool2 = QMaxPool2d(self.block[7], signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool1(x)
        x = self.qconv2(x)
        x = self.qmaxpool2(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool1.freeze(qi=self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qmaxpool2.freeze(qi=self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx = self.qmaxpool1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx, mode=mode)
        qx = self.qmaxpool2.quantize_inference(qx)
        qx = qx.view(-1, 5 * 5 * 64)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class ModelBNNoReLU(nn.Module):
    def __init__(self):
        super(ModelBNNoReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv1 = QConvBNReLU(self.block[0], self.block[1], relu=False, qi=True, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qmaxpool1 = QMaxPool2d(self.block[2], signed=signed)
        self.qconv2 = QConvBNReLU(self.block[3], self.block[4], relu=False, qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qmaxpool2 = QMaxPool2d(self.block[5], signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qmaxpool1(x)
        x = self.qconv2(x)
        x = self.qmaxpool2(x)
        x = x.view(-1, 5 * 5 * 64)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qmaxpool1.freeze(qi=self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qmaxpool2.freeze(qi=self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx = self.qmaxpool1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx, mode=mode)
        qx = self.qmaxpool2.quantize_inference(qx)
        qx = qx.view(-1, 5 * 5 * 64)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class ModelLinear(nn.Module):
    def __init__(self,):
        super(ModelLinear, self).__init__()
        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qlinear1 = QLinearReLU(self.linear1, signed=signed)
        self.qlinear2 = QLinearReLU(self.linear2, qi=False, signed=signed)
        self.qlinear3 = QLinear(self.linear3, qi=False, signed=signed)

    def quantize_forward(self, x):
        x = x.view(-1, 28*28)
        x = self.qlinear1(x)
        x = self.qlinear2(x)
        x = self.qlinear3(x)
        return x

    def freeze(self):
        self.qlinear1.freeze()
        self.qlinear2.freeze(qi=self.qlinear1.qo)
        self.qlinear3.freeze(qi=self.qlinear2.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        x = x.view(-1, 28*28)
        qx = self.qlinear1.qi.quantize_tensor(x)
        qx = self.qlinear1.quantize_inference(qx, mode=mode)
        qx = self.qlinear2.quantize_inference(qx, mode=mode)
        qx = self.qlinear3.quantize_inference(qx, mode=mode)
        out = self.qlinear3.qo.dequantize_tensor(qx)
        return out

class ModelShortCut(nn.Module):
    def __init__(self):
        super(ModelShortCut, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(2)
        self.linear = nn.Linear(32*6*6, 10)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) + x
        x = self.maxpool(x)
        x = x.view(-1, 32*6*6)
        x = self.linear(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv1 = QConvBNReLU(self.block1[0], self.block1[1], qi=True, qo=True,
                num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qconv2 = QConvBNReLU(self.block2[0], self.block2[1], qi=False, qo=True,
                num_bits=num_bits, signed=signed, qmode='per_channel')
        self.qadd = QAdd(qi1=False, qi2=False, qo=True, num_bits=num_bits, signed=signed)
        self.qmaxpool = QMaxPool2d(self.maxpool, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.linear, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x_ = self.qconv2(x)
        x = self.qadd(x_, x)
        x = self.qmaxpool(x)
        x = x.view(-1, 32*6*6)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qadd.freeze(qi1=self.qconv2.qo, qi2=self.qconv2.qi)
        self.qmaxpool.freeze(qi=self.qadd.qo)
        self.qfc.freeze(qi=self.qadd.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx_ = self.qconv2.quantize_inference(qx, mode=mode)
        qx = self.qadd.quantize_inference(qx_, qx, mode=mode)
        qx = self.qmaxpool.quantize_inference(qx)
        qx = qx.view(-1, 32*6*6)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out
    
class ModelLayerNorm(nn.Module):
    def __init__(self):
        super(ModelLayerNorm, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(784, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.block1(x)
        x = self.block2(x)
        x = self.linear(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qlinear1 = QLinear(self.block1[0], qi=True, qo=True, num_bits=num_bits, signed=signed)
        self.qlayernorm1 = QLayerNorm(self.block1[1], qi=False, num_bits=num_bits, signed=signed)
        self.qrelu1 = QReLU(num_bits=num_bits, signed=signed)
        self.qlinear2 = QLinear(self.block2[0], qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.qlayernorm2 = QLayerNorm(self.block2[1], qi=False, num_bits=num_bits, signed=signed)
        self.qrelu2 = QReLU(num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.linear, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = x.view(-1, 28*28)
        x = self.qlinear1(x)
        x = self.qlayernorm1(x, self.qlinear1.qo)
        x = self.qrelu1(x)
        x = self.qlinear2(x)
        x = self.qlayernorm2(x, self.qlinear2.qo)
        x = self.qrelu2(x)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qlinear1.freeze()
        self.qlayernorm1.freeze(qi=self.qlinear1.qo)
        self.qrelu1.freeze(qi=self.qlayernorm1.qo)
        self.qlinear2.freeze(qi=self.qlayernorm1.qo)
        self.qlayernorm2.freeze(qi=self.qlinear2.qo)
        self.qrelu2.freeze(qi=self.qlayernorm2.qo)
        self.qfc.freeze(qi=self.qlayernorm2.qo)
    
    def quantize_inference(self, x, mode='cmsis_nn'):
        x = x.view(-1, 28*28)
        qx = self.qlinear1.qi.quantize_tensor(x)
        qx = self.qlinear1.quantize_inference(qx, mode=mode)
        qx = self.qlayernorm1.quantize_inference(qx, mode=mode)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qlinear2.quantize_inference(qx, mode=mode)
        qx = self.qlayernorm2.quantize_inference(qx, mode=mode)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out
    

class ModelAttention(nn.Module):
    def __init__(self):
        super(ModelAttention, self).__init__()
        self.pre_linear = nn.Linear(28*28, 32)
        self.activation = nn.ReLU()
        self.attn = Attention(dim=32)
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.pre_linear(x)
        x = x.unsqueeze(1)
        x = self.activation(x)
        x = self.attn(x)
        x = self.classifier(x)
        x = x.squeeze(1)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qlinear_relu = QLinearReLU(self.pre_linear, qi=True, num_bits=num_bits, signed=signed)
        self.attn.quantize(first_qi=False)
        self.qclassifier = QLinear(self.classifier, qi=False, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = x.view(-1, 28*28)
        x = self.qlinear_relu(x)
        x = x.unsqueeze(1)
        x = self.attn.quantize_forward(x)
        x = self.qclassifier(x)
        x = x.squeeze(1)
        return x

    def freeze(self):
        self.qlinear_relu.freeze()
        self.attn.freeze(qi=self.qlinear_relu.qo)
        self.qclassifier.freeze(qi=self.attn.qproj.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        x = x.view(-1, 28*28)
        qx = self.qlinear_relu.qi.quantize_tensor(x)
        qx = self.qlinear_relu.quantize_inference(qx, mode=mode)
        qx = qx.unsqueeze(1)
        qx = self.attn.quantize_inference(qx, mode=mode, quantized_input=True)
        qx = self.qclassifier.quantize_inference(qx, mode=mode)
        x = self.qclassifier.qo.dequantize_tensor(qx)
        x = x.squeeze(1)
        return x


class ModelDepthwise(nn.Module):
    def __init__(self,):
        super(ModelDepthwise, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1, groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.fc = nn.Linear(8*14*14, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 8*14*14)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv1 = QConv2d(self.conv[0], qi=True, qo=True, num_bits=num_bits, signed=signed)
        self.qconv2 = QConvBNReLU(self.conv[1], self.conv[2], relu=True, qi=False, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)
    
    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qconv2(x)
        x = x.view(-1, 8*14*14)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx, mode=mode)
        qx = self.qconv2.quantize_inference(qx, mode=mode)
        qx = qx.view(-1, 8*14*14)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class ModelMV2Naive(nn.Module):
    def __init__(self):
        super(ModelMV2Naive, self).__init__()
        self.mv2block2 = MV2Block(1, 8, stride=1, expansion=2)
        self.fc = nn.Linear(8*28*28, 10)

    def forward(self, x):
        x = self.mv2block2(x)
        x = x.view(-1, 8*28*28)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.mv2block2.quantize(first_qi=True, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = self.mv2block2.quantize_forward(x)
        x = x.view(-1, 8*28*28)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.mv2block2.freeze()
        self.qfc.freeze(qi=self.mv2block2.qlast_op.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.mv2block2.qconv[0].qi.quantize_tensor(x)
        qx = self.mv2block2.quantize_inference(qx, mode=mode)
        qx = qx.view(-1, 8*28*28)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


class ModelMV2(nn.Module):
    def __init__(self):
        super(ModelMV2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.mv2block2 = MV2Block(16, 32, stride=2, expansion=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.mv2block2(x)
        x = self.maxpool(x)
        x = x.view(-1, 32*7*7)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv = QConvBNReLU(
            self.conv[0], self.conv[1], relu=True,
            qi=True, num_bits=num_bits, signed=signed
        )
        self.mv2block2.quantize(first_qi=False, num_bits=num_bits, signed=signed)
        self.qmaxpool = QMaxPool2d(self.maxpool, qi=False, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = self.qconv(x)
        x = self.mv2block2.quantize_forward(x)
        x = self.qmaxpool(x)
        x = x.view(-1, 32*7*7)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv.freeze()
        self.mv2block2.freeze(qi=self.qconv.qo)
        self.qmaxpool.freeze(qi=self.mv2block2.qlast_op.qo)
        self.qfc.freeze(qi=self.mv2block2.qlast_op.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv.qi.quantize_tensor(x)
        qx = self.qconv.quantize_inference(qx, mode=mode)
        qx = self.mv2block2.quantize_inference(qx, mode=mode)
        qx = self.qmaxpool.quantize_inference(qx)
        qx = qx.view(-1, 32*7*7)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out

class ModelMV2ShortCut(nn.Module):
    def __init__(self):
        super(ModelMV2ShortCut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.mv2block = MV2Block(16, 16, stride=1, expansion=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(16*7*7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.mv2block(x)
        x = self.maxpool(x)
        x = x.view(-1, 16*7*7)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8, signed=True):
        self.qconv = QConvBNReLU(
            self.conv[0], self.conv[1], relu=True,
            qi=True, num_bits=num_bits, signed=signed
        )
        self.mv2block.quantize(first_qi=False, num_bits=num_bits, signed=signed)
        self.qmaxpool = QMaxPool2d(self.maxpool, qi=False, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed)

    def quantize_forward(self, x):
        x = self.qconv(x)
        x = self.mv2block.quantize_forward(x)
        x = self.qmaxpool(x)
        x = x.view(-1, 16*7*7)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv.freeze()
        self.mv2block.freeze(qi=self.qconv.qo)
        self.qmaxpool.freeze(qi=self.mv2block.qlast_op.qo)
        self.qfc.freeze(qi=self.mv2block.qlast_op.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv.qi.quantize_tensor(x)
        qx = self.qconv.quantize_inference(qx, mode=mode)
        qx = self.mv2block.quantize_inference(qx, mode=mode)
        qx = self.qmaxpool.quantize_inference(qx)
        qx = qx.view(-1, 16*7*7)
        qx = self.qfc.quantize_inference(qx, mode=mode)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out