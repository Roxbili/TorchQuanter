import torch
import torch.nn.functional as F

from .base import QModule, QParam, FakeQuantize, FloorSTE
from torchquanter.utils import quantize_tensor, approximate_float

class QAdaptiveAvgPool2d(QModule):

    def __init__(self, output_size, qi=True, qo=True, num_bits=8, signed=True):
        super(QAdaptiveAvgPool2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.output_size = output_size
        self.num_bits = num_bits
        self.signed = signed

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qi.scale / self.qo.scale
        return self.qo

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.adaptive_avg_pool2d(x, self.output_size)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x
    
    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point
        x = F.adaptive_avg_pool2d(x, self.output_size).floor()
        if mode is None:
            x = self.M * x
            x.round_() 
        elif mode == 'cmsis_nn':
            x = ReScaleAdaptiveAvgPool.apply(x, self.M)
        else:
            raise Exception(f'Unknown mode {mode}')
        x = x + self.qo.zero_point
        x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x


class ReScaleAdaptiveAvgPool(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, M):
        return g.op("ReScale", x, M)

    @staticmethod
    def forward(ctx, x, M):
        multiplier, shift = approximate_float(M)
        round_ = 1 << (shift - 1)
        x = (x * multiplier + round_) >> (31 - shift)
        return x
