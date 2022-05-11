import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QSoftmax
# from models.model import ModelShortCut

torch.manual_seed(0)

class TestSoftmax(nn.Module):
    def __init__(self):
        super(TestSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.softmax(x)
        return x

    def quantize(self):
        self.qsoftmax = QSoftmax(dim=-1, qi=True, qo=True)

    def quantize_forward(self, x):
        x = self.qsoftmax(x)
        return x

    def freeze(self):
        self.qsoftmax.freeze()

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qsoftmax.qi.quantize_tensor(x)
        qx = self.qsoftmax.quantize_inference(qx)
        out = self.qsoftmax.qo.dequantize_tensor(qx)
        return out

def test_qsoftmax():
    data = torch.rand(1,4)

    model = TestSoftmax()
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
    test_qsoftmax()