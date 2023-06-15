import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, FakeQuantize, FloorSTE, QuantizeTensor, DequantizeTensor, RoundSTE
from torchquanter.utils import quantize_tensor, approximate_float

class QSoftmax_W_Policy(QModule):

    def __init__(self, dim=-1, qi=True, qo=True, num_bits=8, signed=True, symmetric_feature=False):
        super(QSoftmax_W_Policy, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed, symmetric=symmetric_feature)
        self.dim = dim
        self.num_bits = num_bits
        self.signed = signed
        self._init_qo(qo)

    def _init_qo(self, qo):
        if qo is True:
            self.qo.scale = torch.tensor(1 / 256., dtype=torch.float32, device=self.qo.scale.device)
            self.qo.zero_point = torch.tensor(-128., dtype=torch.float32, device=self.qo.scale.device)

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')
        self.freeze_flag = True

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        return self.qo
    
    def forward(self, x, policy):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        if self.freeze_flag:
            raise Exception(f'{self._get_name()} has been frozen')
        if policy is None:
            x = F.softmax(x, dim=self.dim)
        else:
            x = self.softmax_with_policy(x, policy)

        # default qo.scale = 1/256, qo.zero_point=-128
        if hasattr(self, 'qo'):
            x = FakeQuantize.apply(x, self.qo)
        
        return x

    def quantize_inference(self, x, policy, **kwargs):
        x = self.qi.dequantize_tensor(x)
        if policy is None:
            x = F.softmax(x, dim=self.dim)
        else:
            x = self.softmax_with_policy(x, policy)
        x = self.qo.quantize_tensor(x)
        return x

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)


'''
# i-bert softmax quant(error) 
# (https://github.com/huggingface/transformers/blob/main/src/transformers/models/ibert/quant_modules.py)

class QSoftmax(QModule):

    def __init__(self, qi=True, qo=True, num_bits=8, signed=True):
        super(QSoftmax, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        assert qo == True, "qo must be True in QSoftmax"
        self.num_bits = num_bits
        self.signed = signed
        self.max_bit = 32
        self.x0 = -0.6931  # -ln2
        self.const = 30  # dummy integer constant
        self.coef = [0.35815147, 0.96963238, 1.0]  # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

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

    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor**2)
        z = (x_int + b_int) * x_int + c_int
        scaling_factor = self.coef[0] * scaling_factor**2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.const * x0_int)

        q = FloorSTE.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(FloorSTE.apply(exp_int * 2 ** (self.const - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**self.const
        return exp_int, scaling_factor

    def forward(self, x, qi=None):
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if hasattr(self, 'qi') and qi is None:  # for test without before_layer.qo
            qi = self.qi
            qi.update(x)

        x_int = QuantizeTensor.apply(x, qi)

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = self.int_exp(x_int, qi.scale)

        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = FloorSTE.apply(2**self.max_bit / exp_int_sum)
        exp_int = FloorSTE.apply(exp_int * factor / 2 ** (self.max_bit - self.num_bits))
        scaling_factor = 1 / 2**self.num_bits

        # update qo
        self.qo.scale = torch.tensor(scaling_factor, dtype=qi.scale.dtype, device=qi.scale.device)
        self.qo.zero_point = torch.tensor(0., dtype=qi.scale.dtype, device=qi.zero_point.device)

        x = DequantizeTensor.apply(exp_int, self.qo)
        return x

    def quantize_inference(self, x, mode=None):
        pass
        # x = x - self.qi.zero_point
        # x = self.fc_module(x)
        # if mode is None:
        #     x = self.M * x
        #     x.round_() 
        # elif mode == 'cmsis_nn':
        #     multiplier, shift = approximate_float(self.M)
        #     x = (x * multiplier) >> (31 - shift)
        # else:
        #     raise Exception(f'Unknown mode {mode}')
        # x = x + self.qo.zero_point
        # x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        # return x
'''