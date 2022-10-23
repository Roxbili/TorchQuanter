import torch.nn.functional as F

from .base import QModule, QParam, FakeQuantize
from torchquanter.utils import quantize_tensor

class QReLU(QModule):

    def __init__(self, qi=False, num_bits=8, signed=True, symmetric_feature=False):
        super(QReLU, self).__init__(qi=qi, qo=False, num_bits=num_bits, signed=signed, symmetric=symmetric_feature)

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

        x = F.relu(x)

        return x
    
    def quantize_inference(self, x, **kwargs):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x