import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .base import QModule, QParamW, FakeQuantize
from torchquanter.utils import quantize_tensor, broadcast_dim_as, approximate_float

class QConvBNReLU(QModule):

    def __init__(self, conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d, relu=True, qi=True, qo=True, 
                 num_bits=8, signed=True, symmetric_weight=True, qmode='per_channel'):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.signed = signed
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.relu = relu
        self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode=qmode)
        # self.qb = QParam(num_bits=32, signed=signed)

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
            
        return weight, bias

    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        if self.training:
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1) # CNHW -> C,NHW
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var
        else:
            mean = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)

        std = torch.sqrt(var + self.bn_module.eps)

        weight, bias = self.fold_bn(mean, std)

        self.qw.update(weight.data)

        x = F.conv2d(x, FakeQuantize.apply(weight, self.qw), bias, 
                stride=self.conv_module.stride,
                padding=self.conv_module.padding, dilation=self.conv_module.dilation, 
                groups=self.conv_module.groups)

        if self.relu:
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

        std = torch.sqrt(self.bn_module.running_var + self.bn_module.eps)

        weight, bias = self.fold_bn(self.bn_module.running_mean, std)
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point.view(-1,1,1,1)

        if self.conv_module.bias is not None:
            self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                         zero_point=0, num_bits=32, signed=True)
        else:
            self.conv_module.bias = nn.Parameter(quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                 zero_point=0, num_bits=32, signed=True), requires_grad=True)

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