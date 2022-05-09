import torch
import torch.nn as nn
from torchquanter.utils import calcScaleZeroPoint, quantize_tensor, dequantize_tensor, get_qmin_qmax
from torch.autograd import Function


class QParam(nn.Module):
    """
    Quantization parameters recorder
    """
    def __init__(self, num_bits=8, signed=True, symmetric=False, momentum=0.9):
        """
        Args
        ----------
        signed: bool, True for int8, False for uint8
        symmetric: bool, True for symmetric quantization(zero_point=0)
        momentum: the value used for the running_mean and running_var computation. Default: 0.9
        """
        super(QParam, self).__init__()
        self.num_bits = num_bits
        self.signed = signed
        self.symmetric = symmetric
        self.momentum = momentum
        self.qmin, self.qmax = get_qmin_qmax(num_bits, signed)

        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)

        # register for saving parameters when calling torch.save API
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)

        # check validity of initial parameters
        self._init_check()

    def _init_check(self):
        assert not (self.signed == False and self.symmetric == True), \
            'Only support symmetirc quantization with signed quantization parameters.'

    def update(self, tensor):
        """
        update the max and min from the tensor,
        calculate the scale and zero point
        """
        raise NotImplementedError('update function is not implemented')
    
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits, signed=self.signed)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        load parameters from state_dict
        """
        pass

    def __str__(self):
        pass


class QParamIO(QParam):
    def __init__(self, num_bits=8, signed=True, symmetric=False, momentum=0.1):
        """
        Args
        ----------
        signed: bool, True for int8, False for uint8
        symmetric: bool, True for symmetric quantization(zero_point=0)
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        """
        super(QParamIO, self).__init__(num_bits=num_bits, signed=signed, symmetric=symmetric, momentum=momentum)

        running_min = torch.tensor([], requires_grad=False)
        running_max = torch.tensor([], requires_grad=False)

        # register for saving parameters when calling torch.save API
        self.register_buffer('running_min', running_min)
        self.register_buffer('running_max', running_max)

    def update(self, tensor):
        """
        update the max and min from the tensor,
        calculate the scale and zero point
        """
        if self.running_max.nelement() == 0:
            self.running_max.data = tensor.max().data
        else:   # exponential moving average update min and max
            self.running_max.data = (1.0 - self.momentum) * self.running_max.data + self.momentum * tensor.max().data
        
        if self.running_min.nelement() == 0:
            self.running_min.data = tensor.min().data
        else:
            self.running_min.data = (1.0 - self.momentum) * self.running_min.data + self.momentum * tensor.min().data
        self.scale, self.zero_point = calcScaleZeroPoint(self.running_min, self.running_max, 
                        self.num_bits, signed=self.signed, symmetric=self.symmetric)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key_names = ['scale', 'zero_point', 'running_min', 'running_max']
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = 'scale: %.10f  ' % self.scale
        info += 'zero_point: %d  ' % self.zero_point
        info += 'running_min: %.6f  ' % self.running_min
        info += 'running_max: %.6f  ' % self.running_max
        return info


class QParamW(QParam):
    def __init__(self, num_bits=8, signed=True, symmetric=True, momentum=0.1, qmode='per_channel'):
        """
        Args
        ----------
        signed: bool, True for int8, False for uint8
        symmetric: bool, True for symmetric quantization(zero_point=0)
        momentum: the value used for the running_mean and running_var computation. Default: 0.1
        qmode: str, per_tensor or per_channel. per_channel quantize along tensor axis 0.
        """
        self.qmode = qmode
        super(QParamW, self).__init__(num_bits=num_bits, signed=signed, symmetric=symmetric, momentum=momentum)

        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)

        # register for saving parameters when calling torch.save API
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def _init_check(self):
        super(QParamW, self)._init_check()
        assert self.qmode in ['per_tensor', 'per_channel'], \
            f"Only 'per_tensor' or 'per_channel' mode is supported"

    def update(self, tensor):
        """
        update the max and min from the tensor,
        calculate the scale and zero point
        """
        if self.qmode == 'per_tensor':
            tensor_max = tensor.max()
            tensor_min = tensor.min()
        elif self.qmode == 'per_channel':
            tensor_max = tensor.flatten(1).max(dim=-1)[0]   # .view(-1, *[1 for _ in range(tensor.dim() - 1)])
            tensor_min = tensor.flatten(1).min(dim=-1)[0]   # .view(-1, *[1 for _ in range(tensor.dim() - 1)])
        else:
            raise Exception("Only 'per_tensor' or 'per_channel' mode is supported")

        if self.max.nelement() == 0:
            self.max.data = tensor_max.data
        else:   # exponential moving average update min and max
            self.max.data = (1.0 - self.momentum) * self.max.data + self.momentum * tensor_max.data
        
        if self.min.nelement() == 0:
            self.min.data = tensor_min.data
        else:   # exponential moving average update min and max
            self.min.data = (1.0 - self.momentum) * self.min.data + self.momentum * tensor_min.data

        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max,
                        self.num_bits, signed=self.signed, symmetric=self.symmetric)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
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
            self.qi = QParamIO(num_bits=num_bits, signed=signed, symmetric=symmetric)
        if qo:
            self.qo = QParamIO(num_bits=num_bits, signed=signed, symmetric=symmetric)

    def freeze(self):
        pass

    def quantize_inference(self, x, mode=None):
        """
        Args
        ----------
        x: float
        mode: None or cmsis_nn. Inference mode, None means use float multiplying. default None.
        """
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