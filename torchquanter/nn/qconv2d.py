import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParamW, FakeQuantize
from torchquanter.utils import quantize_tensor, broadcast_dim_as, approximate_float

class QConv2d(QModule):

    def __init__(self, conv_module: nn.Conv2d, qi=True, qo=True, num_bits=8, 
                 signed=True, symmetric_weight=True, qmode='per_channel'):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.signed = signed
        self.conv_module = conv_module
        self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode=qmode)

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
        self.M = self.qw.scale * self.qi.scale / self.qo.scale

        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point.view(-1,1,1,1)    # 这样减法后可能无法保证范围在 8bit 内

        if self.conv_module.bias is not None:
            self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                         zero_point=0, num_bits=32, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.conv_module.weight.data)    # 统计min、max并计算scale和zero_point

        # 不能使用 x = self.conv_module(x) 的方法，因为修改conv_module.weight会报错，
        # 修改conv_module.data的话无法正常回传梯度(未验证)
        x = F.conv2d(x, FakeQuantize.apply(self.conv_module.weight, self.qw), self.conv_module.bias, 
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                     groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
      
    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        if mode is None:
            x = broadcast_dim_as(self.M, x, dim=1) * x
            x.round_() 
        elif mode == 'cmsis_nn':
            multiplier, shift = approximate_float(self.M)
            round_ = 1 << (shift - 1)
            x = (x * broadcast_dim_as(multiplier, x, dim=1) + broadcast_dim_as(round_, x, dim=1)) \
                    >> (31 - broadcast_dim_as(shift, x, dim=1))
        else:
            raise Exception(f'Unknown mode {mode}')
        x = x + self.qo.zero_point        
        x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x