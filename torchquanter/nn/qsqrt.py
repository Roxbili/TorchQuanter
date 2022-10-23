from math import sqrt
import torch
import torch.nn.functional as F

from .base import DequantizeTensor, QModule, QParam, FakeQuantize, FloorSTE, QuantizeTensor, RoundSTE
from torchquanter.utils import quantize_tensor, approximate_float, sqrt_interger

class QSqrt(QModule):

    def __init__(self, qi=True, qo=True, num_bits=8, signed=True, symmetric_feature=False):
        assert qo == True
        super(QSqrt, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed, symmetric=symmetric_feature)
        self.num_bits = num_bits
        self.signed = signed
        self.first_time = True

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')
        self.freeze_flag = True

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = torch.sqrt(self.qi.scale) / self.qo.scale
        return self.qo

    def forward(self, x, qi=None):
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')

        if hasattr(self, 'qi'):
            qi = self.qi
            qi.update(x)
        if self.freeze_flag:
            raise Exception(f'{self._get_name()} has been frozen')

        if self.first_time:
            out = torch.sqrt(x)
            self.qo.update(out)
            self.first_time=False
        else:
            qx = QuantizeTensor.apply(x, qi)

            qx = qx - qi.zero_point
            qx = FloorSTE.apply(torch.sqrt(qx))
            qx = RoundSTE.apply(torch.sqrt(qi.scale) / self.qo.scale * qx)
            qx = qx + self.qo.zero_point

            x = DequantizeTensor.apply(qx, self.qo)

            if hasattr(self, 'qo'):
                self.qo.update(x)
                x = FakeQuantize.apply(x, self.qo)

        return x
    
    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point
        x = sqrt_interger(x)
        if mode is None:
            x = self.M * x
            x.round_() 
        elif mode == 'cmsis_nn':
            x = ReScaleSqrt.apply(x, self.M)
        else:
            raise Exception(f'Unknown mode {mode}')
        x = x + self.qo.zero_point
        x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x


class ReScaleSqrt(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, M):
        return g.op("ReScale", x, M)

    @staticmethod
    def forward(ctx, x, M):
        multiplier, shift = approximate_float(M)
        round_ = 1 << (shift - 1)
        x = (x * multiplier + round_) >> (31 - shift)
        return x
