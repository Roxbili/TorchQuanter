import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParam, FakeQuantize
from torchquanter.utils import quantize_tensor

class QMaxPool2d(QModule):

    def __init__(self, maxpool2d_module: nn.MaxPool2d, qi=False, num_bits=8, signed=True, symmetric_feature=False):
        super().__init__(qi=qi, num_bits=num_bits, signed=signed, symmetric=symmetric_feature)
        self.maxpool2d_module = maxpool2d_module

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        self.freeze_flag = True

        if qi is not None:
            self.qi = qi
        return self.qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        if self.freeze_flag:
            raise Exception(f'{self._get_name()} has been frozen')

        x = F.max_pool2d(x, self.maxpool2d_module.kernel_size, 
                        self.maxpool2d_module.stride, self.maxpool2d_module.padding)

        return x

    def quantize_inference(self, x, **kwargs):
        return F.max_pool2d(x, self.maxpool2d_module.kernel_size, 
                        self.maxpool2d_module.stride, self.maxpool2d_module.padding)