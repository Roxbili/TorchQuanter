import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParam, FakeQuantize
from torchquanter.utils import quantize_tensor

class QMaxPool2d(QModule):

    def __init__(self, maxpool2d_module: nn.MaxPool2d, qi=False, num_bits=None):
        super().__init__(qi=qi, num_bits=num_bits)
        self.maxpool2d_module = maxpool2d_module

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.max_pool2d(x, self.maxpool2d_module.kernel_size, 
                        self.maxpool2d_module.stride, self.maxpool2d_module.padding)

        return x

    def quantize_inference(self, x):
        return F.max_pool2d(x, self.maxpool2d_module.kernel_size, 
                        self.maxpool2d_module.stride, self.maxpool2d_module.padding)