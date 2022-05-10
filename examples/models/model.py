import torch
import torch.nn as nn
import torch.nn.functional as F

from torchquanter.nn import QConv2d, QMaxPool2d, QReLU, QLinear, QConvBNReLU, QLinearReLU, QAdd, QLayerNorm

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
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')

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
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')

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
        self.qlinear1 = QLinearReLU(self.linear1, signed=signed, qmode='per_channel')
        self.qlinear2 = QLinearReLU(self.linear2, qi=False, signed=signed, qmode='per_channel')
        self.qlinear3 = QLinear(self.linear3, qi=False, signed=signed, qmode='per_channel')

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
        self.qadd = QAdd(qo=True, num_bits=num_bits, signed=signed)
        self.qmaxpool = QMaxPool2d(self.maxpool, num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.linear, qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_tensor')

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
        self.qlinear1 = QLinear(self.block1[0], qi=True, qo=True, num_bits=num_bits, signed=signed, qmode='per_tensor')
        self.qlayernorm1 = QLayerNorm(self.block1[1], qi=False, num_bits=num_bits, signed=signed)
        self.qrelu1 = QReLU(num_bits=num_bits, signed=signed)
        self.qlinear2 = QLinear(self.block2[0], qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_tensor')
        self.qlayernorm2 = QLayerNorm(self.block2[1], qi=False, num_bits=num_bits, signed=signed)
        self.qrelu2 = QReLU(num_bits=num_bits, signed=signed)
        self.qfc = QLinear(self.linear, qi=False, qo=True, num_bits=num_bits, signed=signed, qmode='per_channel')

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
    

if __name__ == '__main__':
    input_data = torch.rand(1, 28, 28, dtype=torch.float32)
    model = Model()

    output = model(input_data)
    print(output.shape)