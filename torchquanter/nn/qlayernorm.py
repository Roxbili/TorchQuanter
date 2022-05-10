import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParamW, FakeQuantize
from torchquanter.utils import quantize_tensor, broadcast_dim_as, approximate_float, sqrt_interger

class QLayerNorm(QModule):

    def __init__(self, layernorm_module: nn.LayerNorm, qi=True, qo=True, num_bits=8, 
                 signed=True, symmetric_weight=True):
        super(QLayerNorm, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.signed = signed
        self.layernorm_module = layernorm_module
        self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode='per_tensor')

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
        self.M = self.qw.scale / self.qo.scale  # 这里非常特殊，没有self.qi.scale，因为输入标准化后完全消除了qi.scale，导致之后无法提取qi.scale了

        self.layernorm_module.weight.data = self.qw.quantize_tensor(self.layernorm_module.weight.data)
        self.layernorm_module.weight.data = self.layernorm_module.weight.data - self.qw.zero_point    # 这样减法后可能无法保证范围在 8bit 内

        self.layernorm_module.bias.data = quantize_tensor(self.layernorm_module.bias.data, scale=self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.layernorm_module.weight.data)    # 统计min、max并计算scale和zero_point

        x = F.layer_norm(x, self.layernorm_module.normalized_shape,
                         weight=FakeQuantize.apply(self.layernorm_module.weight, self.qw), bias=self.layernorm_module.bias,
                         eps=self.layernorm_module.eps)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
      
    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point

        # Interger-only LayerNorm
        mean_ = x.mean(dim=-1, keepdim=True)
        var_ = torch.floor(torch.sum((x - mean_)**2, dim=-1, keepdim=True) / x.shape[-1])
        std_ = sqrt_interger(var_, keepdim=True)
        x_std = torch.floor((x - mean_) / std_)
        if self.layernorm_module.elementwise_affine:
            x = x_std * self.layernorm_module.weight.data + self.layernorm_module.bias.data
        else:
            x = x_std

        if mode is None:
            x = self.M * x
            x.round_() 
        elif mode == 'cmsis_nn':
            multiplier, shift = approximate_float(self.M)
            x = (x * multiplier) >> (31 - shift)
        else:
            raise Exception(f'Unknown mode {mode}')
        x = x + self.qo.zero_point        
        x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x