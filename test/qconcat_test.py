import os, sys
sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn

from torchquanter.nn import QConcat, QConv2d


torch.manual_seed(0)

class TestConcat(nn.Module):
    def __init__(self):
        super(TestConcat, self).__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((out, x), dim=1)
        return out

    def quantize(self):
        self.qconv = QConv2d(self.conv, qi=True, qo=True, qmode='per_channel')
        self.qcat = QConcat(dim=1, qi1=False, qi2=False, qo=True)

    def quantize_forward(self, x):
        x_ = self.qconv(x)
        x = self.qcat(x_, x)
        return x

    def freeze(self):
        self.qconv.freeze()
        self.qcat.freeze(self.qconv.qo, self.qconv.qi)

    def quantize_inference(self, x, mode='cmsis_nn'):
        qx = self.qconv.qi.quantize_tensor(x)
        qx_ = self.qconv.quantize_inference(qx)
        qx = self.qcat.quantize_inference(qx_, qx, mode=mode)
        out = self.qcat.qo.dequantize_tensor(qx)
        return out

def test_qconcat():
    data = torch.rand(1,1,5,5)

    model = TestConcat()
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


if __name__ == '__main__':
    test_qconcat()
