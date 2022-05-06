import torch.nn.functional as F

from .base import QModule, QParam, FakeQuantize
from torchquanter.utils import quantize_tensor

class QReLU(QModule):

    def __init__(self, qi=False, num_bits=8, signed=True):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits, signed=signed)

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

        x = F.relu(x)

        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x