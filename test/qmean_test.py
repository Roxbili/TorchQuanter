import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QMean

torch.manual_seed(0)

class TestMean(nn.Module):
    def __init__(self):
        super(TestMean, self).__init__()
    
    def forward(self, x):
        x = torch.mean(x)
        return x

    def quantize(self):
        self.qmean = QMean(dim=-1, qi=True, qo=True)

    def quantize_forward(self, x):
        x = self.qmean(x)
        return x

    def freeze(self):
        self.qmean.freeze()

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qmean.qi.quantize_tensor(x)
        qx = self.qmean.quantize_inference(qx)
        out = self.qmean.qo.dequantize_tensor(qx)
        return out

def test_mean():
    data = torch.rand(1,4)

    model = TestMean()
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