import torch
import torch.nn as nn
from torchquanter.utils import calcScaleZeroPoint, quantize_tensor, dequantize_tensor, get_qmin_qmax
from torch.autograd import Function


class QParam(nn.Module):
    """
    Quantization parameters recorder
    """
    def __init__(self, num_bits=8, signed=True, symmetric=False):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        self.signed = signed
        self.symmetric = symmetric
        self.qmin, self.qmax = get_qmin_qmax(num_bits, signed)
        assert not (signed == False and symmetric == True), \
            'Only support symmetirc quantization with signed quantization parameters.'

        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)

        # register for saving parameters when calling torch.save API
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def update(self, tensor):
        """
        update the max and min from the tensor,
        calculate the scale and zero point
        """
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)
        
        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)
        
        if self.symmetric:  # symmetric quantization
            self.max.data = max(self.max.abs().data, self.min.abs().data)
            self.min.data = -self.max.data
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits, signed=self.signed)
    
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits, signed=self.signed)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        load parameters from state_dict
        """
        key_names = ['scale', 'zero_point', 'min', 'max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = 'scale: %.10f  ' % self.scale
        info += 'zero_point: %d  ' % self.zero_point
        info += 'min: %.6f  ' % self.min
        info += 'max: %.6f  ' % self.max
        return info


class QModule(nn.Module):
    def __init__(self, qi=True, qo=True, num_bits=8, signed=True, symmetric=False):
        super(QModule, self).__init__()
        if qi:
            self.qi = QParam(num_bits=num_bits, signed=signed, symmetric=symmetric)
        if qo:
            self.qo = QParam(num_bits=num_bits, signed=signed, symmetric=symmetric)

    def freeze(self):
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented.')


class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam: QParam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None