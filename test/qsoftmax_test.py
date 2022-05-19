import os, sys

sys.path.append(os.path.join(os.getcwd(), 'examples/'))
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def test_softmax_s8():
    # 数据来自官方测试用例
    input_data = torch.tensor([-80, -48, 16, 0, -96], dtype=torch.float32)
    gold_output = torch.tensor([-128, -125, 56, -60, -128], dtype=torch.float32)
    input_mult = 1077952576
    input_left_shift = 23
    diff_min = -248 # 暂时不知道干什么用的

    # softmax 不需要input_zero_point，数学上不影响结果
    x = input_data - input_data.max()

    # 这里应该是官方计算中从 int8 -> fixed point 的方法
    x = ((x * input_mult) >> (31 - input_left_shift)) / (1 << (31 - 5))

    # 转成 fixed point后直接输入softmax函数中进行测试，结果正确
    out1 = F.softmax(x, dim=-1)
    out1 = out1 / (1 / 256.) - 128  # output scale和zero_point是定死的
    out1.round_()
    assert (out1 == gold_output).all(), print(out1)


if __name__ == '__main__':
    test_qsoftmax()
    test_softmax_s8()