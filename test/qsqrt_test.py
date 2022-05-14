import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QSqrt

torch.manual_seed(0)

class TestSqrt(nn.Module):
    def __init__(self):
        super(TestSqrt, self).__init__()
    
    def forward(self, x):
        x = torch.sqrt(x)
        return x

    def quantize(self):
        self.qsqrt = QSqrt()

    def quantize_forward(self, x):
        x = self.qsqrt(x)
        return x

    def freeze(self):
        self.qsqrt.freeze()

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qsqrt.qi.quantize_tensor(x)
        qx = self.qsqrt.quantize_inference(qx, mode=mode)
        out = self.qsqrt.qo.dequantize_tensor(qx)
        return out

def test_mean():
    data = torch.rand(1,10)

    model = TestSqrt()
    model(data)
    out = model(data).flatten()

    model.eval()
    model.quantize()
    for _ in range(10):
        simulate_out = model.quantize_forward(data)
    model.freeze()

    qout_float = model.quantize_inference(data).flatten()
    err = (out - qout_float).abs().mean()
    assert err < 0.1, f'err: {err}'

if __name__ == '__main__':
    test_mean()