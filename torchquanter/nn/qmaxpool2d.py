from .base import QModule, QParam
from torchquanter.utils import quantize_tensor

class QMaxPool2d(QModule):

    def __init__(self, maxpool2d_module, qi=False, num_bits=None):
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

        x = self.maxpool2d_module(x)

        return x

    def quantize_inference(self, x):
        return self.maxpool2d_module(x)