import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, FakeQuantize, QParamIO
from torchquanter.utils import broadcast_dim_as, approximate_float

class QSub(QModule):

    def __init__(self, qi1=True, qi2=True, qo=True, num_bits=8, signed=True):
        """
        return
        ----------
        output: x1 - x2
        """
        super(QSub, self).__init__(qi=False, qo=qo, num_bits=num_bits, signed=signed)
        if qi1:
            self.qi1 = QParamIO(num_bits=num_bits, signed=signed, symmetric=False)
        if qi2:
            self.qi2 = QParamIO(num_bits=num_bits, signed=signed, symmetric=False)
        self.num_bits = num_bits
        self.signed = signed

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        M1 = torch.tensor([], requires_grad=False, device=device)
        self.register_buffer('M1', M1)
        M2 = torch.tensor([], requires_grad=False, device=device)
        self.register_buffer('M2', M2)

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
        self.M1 = self.qi1.scale / self.qo.scale
        self.M2 = self.qi2.scale / self.qo.scale

    def forward(self, x1, x2):
        if hasattr(self, 'qi1'):
            self.qi1.update(x1)
            x1 = FakeQuantize.apply(x1, self.qi1)
        if hasattr(self, 'qi2'):
            self.qi2.update(x2)
            x2 = FakeQuantize.apply(x2, self.qi2)

        out = x1 - x2

        if hasattr(self, 'qo'):
            self.qo.update(out)
            out = FakeQuantize.apply(out, self.qo)

        return out
      
    def quantize_inference(self, x1, x2, mode=None):
        x1 = x1 - self.qi1.zero_point
        x2 = x2 - self.qi2.zero_point
        if mode is None:
            x1 = self.M1 * x1
            x2 = self.M2 * x2
            out = x1 - x2
            out.round_() 
        elif mode == 'cmsis_nn':
            multiplier1, shift1 = approximate_float(self.M1)
            round1 = 1 << (shift1 - 1)
            multiplier2, shift2 = approximate_float(self.M2)
            round2 = 1 << (shift2 - 1)

            x1 = (x1 * multiplier1 + round1) >> (31 - shift1)
            x2 = (x2 * multiplier2 + round2) >> (31 - shift2)
            out = x1 - x2
        else:
            raise Exception(f'Unknown mode {mode}')
        out = out + self.qo.zero_point
        out.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return out