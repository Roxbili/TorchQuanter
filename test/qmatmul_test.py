import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QLinear, QMatmul

torch.manual_seed(0)

class TestMul(nn.Module):
    def __init__(self):
        super(TestMul, self).__init__()
        self.fc = nn.Linear(2, 4)
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.matmul(x, x.permute(1,0))
        return x

    def quantize(self):
        self.qfc = QLinear(self.fc, qi=True, qo=True)
        self.qmatmul = QMatmul(qi1=False, qi2=False)

    def quantize_forward(self, x):
        x = self.qfc(x)
        x = self.qmatmul(x, x.permute(1,0))
        return x

    def freeze(self):
        self.qfc.freeze()
        self.qmatmul.freeze(self.qfc.qo, self.qfc.qo)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qfc.qi.quantize_tensor(x)
        qx = self.qfc(qx)
        qx = self.qmatmul(qx, qx.permute(1,0))
        out = self.qmatmul.qo.dequantize_tensor(qx)
        return out

def test_qadd1():
    data = torch.rand(10,2)

    model = TestMul()
    model(data)
    out = model(data).flatten()

    model.eval()
    model.quantize()
    for _ in range(10):
        model.quantize_forward(data)
    model.freeze()

    qout_float = model.quantize_inference(data, mode=None).flatten()
    err = (out - qout_float).abs().mean()
    assert err < 0.3, f'err: {err}'


if __name__ == '__main__':
    test_qadd1()