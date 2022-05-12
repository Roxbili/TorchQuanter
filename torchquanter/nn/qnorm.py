import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParamW, FakeQuantize, FloorSTE, QuantizeTensor, DequantizeTensor, RoundSTE, ClampSTE
from torchquanter.utils import quantize_tensor, broadcast_dim_as, approximate_float, sqrt_interger, get_qmin_qmax

class QNorm(QModule):
    """
    (x - mean) / var
    """

    def __init__(self, qi=True, qo=True, num_bits=8, max_bits=32, signed=True):
        super(QNorm, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.max_bits = max_bits
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

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M = 1 / self.qo.scale

    def forward(self, x, qi=None):
        """
        here need to get before_layer.qo as norm.qi
        """
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if hasattr(self, 'qi') and qi is not None:  # for test without before_layer.qo
            raise ValueError('qi has been provided in init function.')
        if hasattr(self, 'qi') and qi is None:  # for test without before_layer.qo
            qi = self.qi
            qi.update(x)

        qx = QuantizeTensor.apply(x, qi)
        qx = qx - qi.zero_point

        # Interger-only Norm
        mean_ = qx.mean(dim=-1, keepdim=True)
        sum_ = ClampSTE.apply(torch.sum((qx - mean_)**2, dim=-1, keepdim=True), 
                    *get_qmin_qmax(self.max_bits, signed=True)) # int32
        var_ = FloorSTE.apply(sum_ / qx.shape[-1])
        std_ = FloorSTE.apply(torch.sqrt(var_))
        x = FloorSTE.apply(((qx - mean_) / std_))   # float32

        self.qo.update(x)
        x = FakeQuantize.apply(x, self.qo)

        return x
      
    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point

        # Interger-only LayerNorm
        mean_ = x.mean(dim=-1, keepdim=True)    # int16
        sum_ = torch.sum((x - mean_)**2, dim=-1, keepdim=True).clamp(*get_qmin_qmax(self.max_bits, signed=True))    # 裁剪到32bit范围内
        var_ = torch.floor(sum_ / x.shape[-1])
        std_ = sqrt_interger(var_, keepdim=True)
        x = torch.floor((x - mean_) / std_)

        if mode is None:
            x = self.M * x
            x.round_() 
        elif mode == 'cmsis_nn':
            multiplier, shift = approximate_float(self.M)
            round_ = 1 << (shift - 1)
            x = (x * multiplier + round_) >> (31 - shift)
        else:
            raise Exception(f'Unknown mode {mode}')
        x = x + self.qo.zero_point        
        x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x