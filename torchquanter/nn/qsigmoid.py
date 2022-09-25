import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, FakeQuantize, FloorSTE, QuantizeTensor, DequantizeTensor, RoundSTE
from torchquanter.utils import quantize_tensor, approximate_float

class QSigmoid(QModule):

    def __init__(self, qi=True, qo=True, num_bits=8, signed=True):
        super(QSigmoid, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.signed = signed

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

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.sigmoid(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def quantize_inference(self, x):
        x = self.qi.dequantize_tensor(x)
        x = F.sigmoid(x)
        x = self.qo.quantize_tensor(x)
        return x

