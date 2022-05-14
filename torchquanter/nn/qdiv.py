import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, FakeQuantize, QParamIO, FloorSTE, QuantizeTensor, DequantizeTensor
from torchquanter.utils import broadcast_dim_as, approximate_float

class QDiv(QModule):
    # High error
    pass

    '''
    def __init__(self, mul_const=None, qi1=True, qi2=True, qo=True, num_bits=8, signed=True):
        """
        Args
        ----------
        mul_const: if not None, x1 / x2 * mul_const
        """
        super(QDiv, self).__init__(qi=False, qo=qo, num_bits=num_bits, signed=signed)
        if qi1:
            self.qi1 = QParamIO(num_bits=num_bits, signed=signed, symmetric=False)
        if qi2:
            self.qi2 = QParamIO(num_bits=num_bits, signed=signed, symmetric=False)
        self.mul_const = mul_const if mul_const is not None else 1.
        self.num_bits = num_bits
        self.signed = signed
        self.first_time = True

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
        self.M = self.qi1.scale * self.mul_const / (self.qo.scale * self.qi2.scale)

    def forward(self, x1, x2, qi1=None, qi2=None):
        """
        here need to get before_layer.qo as here.qi
        """
        if not hasattr(self, 'qi1') and qi1 is None:
            raise ValueError('qi1 is not existed, should be provided.')
        if hasattr(self, 'qi1') and qi1 is not None:
            raise ValueError('qi1 has been provided in init function.')
        if not hasattr(self, 'qi2') and qi2 is None:
            raise ValueError('qi2 is not existed, should be provided.')
        if hasattr(self, 'qi2') and qi2 is not None:
            raise ValueError('qi2 has been provided in init function.')

        if hasattr(self, 'qi1'):
            qi1 = self.qi1
            qi1.update(x1)
            x1 = FakeQuantize.apply(x1, self.qi1)
        if hasattr(self, 'qi2'):
            qi2 = self.qi2
            qi2.update(x2)
            x1 = FakeQuantize.apply(x1, self.qi2)

        if self.first_time:
            out = torch.div(x1, x2)
            self.qo.update(out)
            self.first_time=False
        else:
            qx1 = QuantizeTensor.apply(x1, qi1)
            qx2 = QuantizeTensor.apply(x2, qi2)

            qx1 = qx1 - qi1.zero_point
            qx2 = qx2 - qi2.zero_point

            qx = FloorSTE.apply(qx1 / qx2)
            out = self.mul_const * qi1.scale * qx / (self.qo.scale * qi2.scale)

            if hasattr(self, 'qo'):
                self.qo.update(out)
                out = FakeQuantize.apply(out, self.qo)

        return out
      
    def quantize_inference(self, x1, x2, mode=None):
        x1 = x1 - self.qi1.zero_point
        x2 = x2 - self.qi2.zero_point
        if mode is None:
            out = torch.div(x1, x2).floor()
            out = out * self.M
            out.round_() 
        elif mode == 'cmsis_nn':
            multiplier, shift = approximate_float(self.M)
            round_ = 1 << (shift - 1)
            out = torch.div(x1, x2).floor()
            out = (out * multiplier + round_) >> (31 - shift)
        else:
            raise Exception(f'Unknown mode {mode}')
        out = out + self.qo.zero_point        
        out.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return out
    '''