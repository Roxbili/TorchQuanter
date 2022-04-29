import torch
import torch.nn as nn

from torchquanter.nn import QConv2d, QMaxPool2d, QReLU, QLinear

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

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.block[0], qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU(qi=False)
        self.qmaxpool1 = QMaxPool2d(self.block[2], qi=False)
        self.qconv2 = QConv2d(self.block[3], qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU(qi=False)
        self.qmaxpool2 = QMaxPool2d(self.block[5], qi=False)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

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

    def quantize_inference(self, x):
        """
        真正的量化推理，使用int8
        """
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2.quantize_inference(qx)
        qx = qx.view(-1, 5 * 5 * 64)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out


if __name__ == '__main__':
    input_data = torch.rand(1, 28, 28, dtype=torch.float32)
    model = Model()

    output = model(input_data)
    print(output.shape)