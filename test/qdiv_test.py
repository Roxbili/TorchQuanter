import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QDiv

torch.manual_seed(0)

class TestDiv(nn.Module):
    def __init__(self):
        super(TestDiv, self).__init__()
    
    def forward(self, x1, x2):
        x = torch.div(x1, x2)
        return x

    def quantize(self):
        self.qdiv = QDiv()

    def quantize_forward(self, x1, x2):
        x = self.qdiv(x1, x2)
        return x

    def freeze(self):
        self.qdiv.freeze()

    def quantize_inference(self, x1, x2, mode='cmsis_nn'):
        qx1 = self.qdiv.qi1.quantize_tensor(x1)
        qx2 = self.qdiv.qi2.quantize_tensor(x2)
        qx = self.qdiv.quantize_inference(qx1, qx2, mode=mode)
        out = self.qdiv.qo.dequantize_tensor(qx)
        return out

def test_mean():
    data1 = torch.rand(1,10)
    data2 = torch.rand(1,10)

    model = TestDiv()
    model(data1, data2)
    out = model(data1, data2).flatten()

    model.eval()
    model.quantize()
    for _ in range(10):
        simulate_out = model.quantize_forward(data1, data2)
    model.freeze()

    qout_float = model.quantize_inference(data1, data2).flatten()
    err = (out - qout_float).abs().mean()
    assert err < 0.1, f'err: {err}'

if __name__ == '__main__':
    test_mean()