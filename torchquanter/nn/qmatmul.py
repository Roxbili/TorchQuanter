import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, FakeQuantize, QParamIO
from torchquanter.utils import broadcast_dim_as, approximate_float

class QMatmul(QModule):
    """
    Dot produc function
    """

    def __init__(self, qi1=True, qi2=True, qo=True, mul_const=None, num_bits=8, signed=True):
        """
        Args
        ----------
        const: if not None, torch.matmul(x1, x2) * const
        """
        super(QMatmul, self).__init__(qi=False, qo=qo, num_bits=num_bits, signed=signed)
        if qi1:
            self.qi1 = QParamIO(num_bits=num_bits, signed=signed, symmetric=False)
        if qi2:
            self.qi2 = QParamIO(num_bits=num_bits, signed=signed, symmetric=False)
        self.mul_const = mul_const if mul_const is not None else 1.
        self.num_bits = num_bits
        self.signed = signed

    def freeze(self, qi1=None, qi2=None, qo=None):
        
        if hasattr(self, 'qi1') and qi1 is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi1') and qi1 is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qi2') and qi2 is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi2') and qi2 is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi1 is not None:
            self.qi1 = qi1
        if qi2 is not None:
            self.qi2 = qi2
        if qo is not None:
            self.qo = qo
        self.M = self.qi1.scale * self.qi2.scale * self.mul_const / self.qo.scale

    def forward(self, x1, x2):
        if hasattr(self, 'qi1'):
            self.qi1.update(x1)
            x1 = FakeQuantize.apply(x1, self.qi1)
        if hasattr(self, 'qi2'):
            self.qi2.update(x2)
            x2 = FakeQuantize.apply(x2, self.qi2)

        out = torch.matmul(x1, x2) * self.mul_const

        if hasattr(self, 'qo'):
            self.qo.update(out)
            out = FakeQuantize.apply(out, self.qo)

        return out
      
    def quantize_inference(self, x1, x2, mode=None):
        x1 = x1 - self.qi1.zero_point
        x2 = x2 - self.qi2.zero_point
        if mode is None:
            out = torch.matmul(x1, x2)
            out = out * self.M
            out.round_() 
        elif mode == 'cmsis_nn':
            out = ReScaleMatMul.apply(x1, x2, self.M)
        else:
            raise Exception(f'Unknown mode {mode}')
        out = out + self.qo.zero_point        
        out.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return out


class ReScaleMatMul(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x1, x2, M):
        return g.op("ReScale", x1, x2, M)

    @staticmethod
    def forward(ctx, x1, x2, M):
        multiplier, shift = approximate_float(M)
        round_ = 1 << (shift - 1)
        out = torch.matmul(x1, x2)
        out = (out * multiplier + round_) >> (31 - shift)
        return out
