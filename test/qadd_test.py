import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QAdd, QConv2d
from models.model import ModelShortCut


torch.manual_seed(0)

class TestAdd(nn.Module):
    def __init__(self):
        super(TestAdd, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv(x) + x
        return x

    def quantize(self):
        self.qconv = QConv2d(self.conv, qi=True, qo=True, qmode='per_channel')
        self.qadd = QAdd(qo=True)

    def quantize_forward(self, x):
        x_ = self.qconv(x)
        x = self.qadd(x_, x)
        return x

    def freeze(self):
        self.qconv.freeze()
        self.qadd.freeze(self.qconv.qo, self.qconv.qi)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv.qi.quantize_tensor(x)
        qx_ = self.qconv.quantize_inference(qx)
        qx = self.qadd.quantize_inference(qx_, qx, mode=mode)
        out = self.qadd.qo.dequantize_tensor(qx)
        return out

def test_qadd1():
    data = torch.rand(1,1,5,5)

    model = TestAdd()
    model(data)
    out = model(data).flatten()

    model.eval()
    model.quantize()
    for _ in range(10):
        model.quantize_forward(data)
    model.freeze()

    qout_float = model.quantize_inference(data, mode=None).flatten()
    err = (out - qout_float).abs().mean()
    assert err < 0.1, f'err: {err}'

def test_qadd2():
    data = torch.rand(1,1,28,28)
    model = ModelShortCut()
    out = model(data).flatten()

    model.eval()
    model.quantize()
    for _ in range(10):
        model.quantize_forward(data)
    model.freeze()

    qout_float = model.quantize_inference(data).flatten()
    err = (out - qout_float).abs().mean()
    assert err < 0.1, f'err: {err}'


if __name__ == '__main__':
    test_qadd1()
    # test_qadd2()