import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParamW, FakeQuantize
from torchquanter.utils import quantize_tensor, approximate_float

class QLinearReLU(QModule):

    def __init__(self, fc_module: nn.Linear, qi=True, qo=True, num_bits=8,
                 signed=True, symmetric_weight=True, qmode='per_channel'):
        super(QLinearReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.signed = signed
        self.fc_module = fc_module
        self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode=qmode)
        # self.qb = QParam(num_bits=32)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw), self.fc_module.bias)
        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

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

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point.view(-1,1)

        self.fc_module.bias.data = quantize_tensor(self.fc_module.bias.data, scale=self.qi.scale * self.qw.scale,
                                                   zero_point=0, num_bits=32, signed=True)

    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
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
