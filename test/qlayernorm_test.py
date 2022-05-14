import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QLayerNorm, QLayerNormTFLite
# from models.model import ModelShortCut

torch.manual_seed(0)

class TestQlayernorm(nn.Module):
    def __init__(self):
        super(TestQlayernorm, self).__init__()
        self.layernorm = nn.LayerNorm(10)
    
    def forward(self, x):
        x = self.layernorm(x)
        return x

    def quantize(self):
        self.qlayernorm = QLayerNorm(self.layernorm, qi=True, qo=True)
        # self.qlayernorm = QLayerNormTFLite(self.layernorm, qi=True, qo=True)

    def quantize_forward(self, x):
        x = self.qlayernorm(x)
        return x

    def freeze(self):
        self.qlayernorm.freeze()

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qlayernorm.qi.quantize_tensor(x)
        qx = self.qlayernorm.quantize_inference(qx, mode=mode)
        out = self.qlayernorm.qo.dequantize_tensor(qx)
        return out

def test_qlayernorm():
    data = torch.rand(1,10)

    model = TestQlayernorm()
    model(data)
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
    test_qlayernorm()