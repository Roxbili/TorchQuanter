from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QModule, QParamW, FakeQuantize, FloorSTE, QuantizeTensor, DequantizeTensor, RoundSTE, ClampSTE
from .qnorm import QNorm
from .qmean import QMean
from .qadd import QAdd
from .qsub import QSub
from .qmul import QMul
from .qsqrt import QSqrt
from .qdiv import QDiv
from torchquanter.utils import quantize_tensor, broadcast_dim_as, approximate_float, sqrt_interger, get_qmin_qmax

#### Deprecated!
# class QLayerNorm(QModule):
#     """
#     QNorm * weight + bias
#     """

#     def __init__(self, layernorm_module: nn.LayerNorm, qi=True, qo=True, num_bits=8, max_bits=32,
#                  signed=True, symmetric_weight=True):
#         qlayernorm_qo = qo if layernorm_module.elementwise_affine else False    # affine为False则直接使用QNorm的qo即可
#         super(QLayerNorm, self).__init__(qi=qi, qo=qlayernorm_qo, num_bits=num_bits, signed=signed)
#         self.num_bits = num_bits
#         self.max_bits = max_bits
#         self.signed = signed
#         self.layernorm_module = layernorm_module
#         self.qnorm = QNorm(qi=False, qo=True, num_bits=num_bits, max_bits=32, signed=signed)

#         if self.layernorm_module.elementwise_affine:
#             self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode='per_tensor')

#     def freeze(self, qi=None, qo=None):
        
#         if hasattr(self, 'qi') and qi is not None:
#             raise ValueError('qi has been provided in init function.')
#         if not hasattr(self, 'qi') and qi is None:
#             raise ValueError('qi is not existed, should be provided.')

#         if hasattr(self, 'qo') and qo is not None:
#             raise ValueError('qo has been provided in init function.')

#         if qi is not None:
#             self.qi = qi
#         if qo is not None:
#             self.qo = qo

#         self.qnorm.freeze(qi=self.qi)

#         if self.layernorm_module.elementwise_affine:
#             self.M = self.qnorm.qo.scale * self.qw.scale / self.qo.scale  # 这里非常特殊，没有self.qi.scale，因为输入标准化后完全消除了qi.scale，导致之后无法提取qi.scale了

#             self.layernorm_module.weight.data = self.qw.quantize_tensor(self.layernorm_module.weight.data)
#             self.layernorm_module.weight.data = self.layernorm_module.weight.data - self.qw.zero_point    # 这样减法后可能无法保证范围在 8bit 内

#             self.layernorm_module.bias.data = quantize_tensor(self.layernorm_module.bias.data, scale=self.qnorm.qo.scale * self.qw.scale,
#                                                         zero_point=0, num_bits=32, signed=True)
#         else:
#             self.qo = self.qnorm.qo

#     def forward(self, x, qi=None):
#         """
#         here need to get before_layer.qo as layernorm.qi
#         """
#         if not hasattr(self, 'qi') and qi is None:
#             raise ValueError('qi is not existed, should be provided.')
#         if hasattr(self, 'qi') and qi is not None:  # for test without before_layer.qo
#             raise ValueError('qi has been provided in init function.')
#         if hasattr(self, 'qi') and qi is None:  # for test without before_layer.qo
#             qi = self.qi
#             qi.update(x)

#         x = self.qnorm(x, qi)

#         if self.layernorm_module.elementwise_affine:
#             self.qw.update(self.layernorm_module.weight.data)    # 统计min、max并计算scale和zero_point
#             x = torch.mul(x, FakeQuantize.apply(self.layernorm_module.weight, self.qw)) + self.layernorm_module.bias

#         if hasattr(self, 'qo'):
#             self.qo.update(x)
#             x = FakeQuantize.apply(x, self.qo)
            
#         return x
      
#     def quantize_inference(self, x, mode=None):
#         x = self.qnorm.quantize_inference(x, mode=mode)

#         if self.layernorm_module.elementwise_affine:
#             x = x - self.qnorm.qo.zero_point

#             x = x * self.layernorm_module.weight.data + self.layernorm_module.bias.data

#             if mode is None:
#                 x = self.M * x
#                 x.round_() 
#             elif mode == 'cmsis_nn':
#                 multiplier, shift = approximate_float(self.M)
#                 round_ = 1 << (shift - 1)
#                 x = (x * multiplier + round_) >> (31 - shift)
#             else:
#                 raise Exception(f'Unknown mode {mode}')
#             x = x + self.qo.zero_point        
#             x.clamp_(self.qo.qmin, self.qo.qmax).round_()
#         return x


class QLayerNorm(QModule):
    """
    整体量化
    """

    def __init__(self, layernorm_module: nn.LayerNorm, qi=True, qo=True, num_bits=8, max_bits=32,
                 signed=True, symmetric_weight=True):
        super(QLayerNorm, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.max_bits = max_bits
        self.layernorm_module = layernorm_module
        self.signed = signed
        self.scale = 2**(8 - 1)
        self.first_time = True

        if self.layernorm_module.elementwise_affine:
            self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode='per_tensor')

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

        if self.layernorm_module.elementwise_affine:
            self.M = self.qw.scale / (self.qo.scale * self.scale)  # 这里非常特殊，没有self.qi.scale，因为输入标准化后完全消除了qi.scale，导致之后无法提取qi.scale了
            self.layernorm_module.weight.data = self.qw.quantize_tensor(self.layernorm_module.weight.data)
            self.layernorm_module.weight.data = self.layernorm_module.weight.data - self.qw.zero_point    # 这样减法后可能无法保证范围在 8bit 内

            self.layernorm_module.bias.data = quantize_tensor(self.layernorm_module.bias.data, scale=self.qw.scale / self.scale,
                                                        zero_point=0, num_bits=32, signed=True)
        else:
            self.M = 1 / (self.qo.scale * self.scale)  # 这里非常特殊，没有self.qi.scale，因为输入标准化后完全消除了qi.scale，导致之后无法提取qi.scale了
        return self.qo

    def forward(self, x, qi=None):
        """
        here need to get before_layer.qo as norm.qi
        """
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if hasattr(self, 'qi') and qi is not None:  # for test without before_layer.qo
            raise ValueError('qi has been provided in init function.')
        if hasattr(self, 'qi') and qi is None:  # for test without before_layer.qo
            qi = self.qi
            qi.update(x)

        if self.qo.scale.numel() == 0: 
            x = F.layer_norm(x, self.layernorm_module.normalized_shape,
                    self.layernorm_module.weight, self.layernorm_module.bias,
                    self.layernorm_module.eps)
        else:
            qx = QuantizeTensor.apply(x, qi)
            qx = qx - qi.zero_point

            # Interger-only Norm
            mean_ = qx.mean(dim=-1, keepdim=True).clamp(*get_qmin_qmax(16, signed=True))
            sum_ = ClampSTE.apply(torch.sum((qx - mean_)**2, dim=-1, keepdim=True), 
                        *get_qmin_qmax(self.max_bits, signed=True)) # int32, 这里超出去直接裁剪可能不如偏移来得好，先这么做吧
            var_ = FloorSTE.apply(sum_ / qx.shape[-1])
            var_[var_ == 0.] = 1.   # prevent overflow
            std_ = FloorSTE.apply(torch.sqrt(var_))
            factor = FloorSTE.apply(2**(8 - 1) / std_)
            qx = FloorSTE.apply(ClampSTE.apply((qx - mean_) * factor, *get_qmin_qmax(16, signed=True)))
            x = qx / 2**(8 - 1) # 不需要floor因为这个除法是整合到M中去的

            if self.layernorm_module.elementwise_affine:
                self.qw.update(self.layernorm_module.weight.data)    # 统计min、max并计算scale和zero_point
                x = torch.mul(x, FakeQuantize.apply(self.layernorm_module.weight, self.qw)) + self.layernorm_module.bias
                x = x.clamp(*get_qmin_qmax(self.max_bits, signed=True))

        self.qo.update(x)
        x = FakeQuantize.apply(x, self.qo)
        return x
      
    def quantize_inference(self, x, mode=None):
        x = x - self.qi.zero_point

        if self.layernorm_module.elementwise_affine:
            x = QLayerNormAffine.apply(
                x,
                self.layernorm_module.weight,
                self.layernorm_module.bias,
                self.layernorm_module.eps,
                list(range(-1, -len(self.layernorm_module.normalized_shape) - 1, -1)),
                self.scale,
                self.max_bits
            )
        else:
            x = QLayerNormNoAffine.apply(
                x,
                self.layernorm_module.eps,
                list(range(-1, -len(self.layernorm_module.normalized_shape) - 1, -1)),
                self.scale,
                self.max_bits
            )

        if mode is None:
            x = self.M * x
            x.round_() 
        elif mode == 'cmsis_nn':
            x = ReScaleLayerNorm.apply(x, self.M)
        else:
            raise Exception(f'Unknown mode {mode}')
        x = x + self.qo.zero_point
        x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x


class ReScaleLayerNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, M):
        return g.op("ReScale", x, M)

    @staticmethod
    def forward(ctx, x, M):
        multiplier, shift = approximate_float(M)
        round_ = 1 << (shift - 1)
        x = (x * multiplier + round_) >> (31 - shift)
        return x


class QLayerNormAffine(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, weight, bias, eps, axis, scale, max_bits):
        return g.op(
            "QLayerNorm",
            x,
            weight,
            bias,
            epsilon_f=eps,
            axis_i=axis,
            scale_i=scale,
            max_bits_i=max_bits
        )

    @staticmethod
    def forward(ctx, x, weight, bias, eps, axis, scale, max_bits):
        # Interger-only LayerNorm
        mean_ = x.mean(dim=-1, keepdim=True).clamp(*get_qmin_qmax(16, signed=True))    # int16
        mean_ = mean_.round()
        sum_ = torch.sum((x - mean_)**2, dim=-1, keepdim=True).clamp(*get_qmin_qmax(max_bits, signed=True))    # 裁剪到32bit范围内
        var_ = torch.floor(sum_ / x.shape[-1])
        var_[var_ == 0.] = 1.   # prevent overflow
        # std_ = sqrt_interger(var_)  # 比较费时间，此处快速评估无需使用
        std_ = torch.sqrt(var_).floor()
        factor = torch.floor(scale / std_)
        x = torch.floor(torch.clamp((x - mean_) * factor, *get_qmin_qmax(16, signed=True)))

        x = x * weight.data + bias.data
        x = x.clamp(*get_qmin_qmax(max_bits, signed=True))
        return x


class QLayerNormNoAffine(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, eps, axis, scale, max_bits):
        return g.op(
            "QLayerNorm",
            x,
            epsilon_f=eps,
            axis_i=axis,
            scale_i=scale,
            max_bits_i=max_bits
        )

    @staticmethod
    def forward(ctx, x, eps, axis, scale, max_bits):
        # Interger-only LayerNorm
        mean_ = x.mean(dim=-1, keepdim=True).clamp(*get_qmin_qmax(16, signed=True))    # int16
        mean_ = mean_.round()
        sum_ = torch.sum((x - mean_)**2, dim=-1, keepdim=True).clamp(*get_qmin_qmax(max_bits, signed=True))    # 裁剪到32bit范围内
        var_ = torch.floor(sum_ / x.shape[-1])
        var_[var_ == 0.] = 1.   # prevent overflow
        # std_ = sqrt_interger(var_)  # 比较费时间，此处快速评估无需使用
        std_ = torch.sqrt(var_).floor()
        factor = torch.floor(scale / std_)
        x = torch.floor(torch.clamp((x - mean_) * factor, *get_qmin_qmax(max_bits, signed=True)))
        return x



class QLayerNormFP32(QModule):
    """
    使用FP32进行计算
    """

    def __init__(self, layernorm_module: nn.LayerNorm, qi=True, qo=True, num_bits=8, max_bits=32,
                 signed=True, symmetric_weight=True):
        super(QLayerNormFP32, self).__init__(qi=qi, qo=qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.max_bits = max_bits
        self.layernorm_module = layernorm_module
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

    def forward(self, x, qi=None):
        """
        qi=None是为了保证和之前一致，使得代码可以服用
        """
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.layernorm_module(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        return x

    def quantize_inference(self, x, mode=None):
        x = self.qi.dequantize_tensor(x)
        x = self.layernorm_module(x)
        x = self.qo.quantize_tensor(x)
        return x


class QLayerNormTFLite(QModule):
    pass
    '''
    # High error
    def __init__(self, layernorm_module: nn.LayerNorm, qi=True, qo=True, num_bits=8, max_bits=32,
                 signed=True, symmetric_weight=True):
        qlayernorm_qo = qo if layernorm_module.elementwise_affine else False    # affine为False则直接使用QMul的qo即可
        super(QLayerNormTFLite, self).__init__(qi=qi, qo=qlayernorm_qo, num_bits=num_bits, signed=signed)
        self.num_bits = num_bits
        self.max_bits = max_bits
        self.signed = signed
        self.layernorm_module = layernorm_module

        # numerator
        self.mean = QMean(dim=-1, keepdim=True, qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.sub = QSub(qi1=False, qi2=False, qo=True, num_bits=num_bits, signed=signed)

        # denominator
        self.var_mul = QMul(qi1=False, qi2=False, qo=True, num_bits=num_bits, signed=signed)
        self.var_mean = QMean(dim=-1, keepdim=True, qi=False, qo=True, num_bits=num_bits, signed=signed)
        self.var_sqrt = QSqrt(qi=False, qo=True, num_bits=num_bits, signed=signed)

        # numerator / denominator
        self.div = QDiv(qi1=False, qi2=False, qo=True, num_bits=num_bits, signed=signed)

        if self.layernorm_module.elementwise_affine:
            self.qw = QParamW(num_bits=num_bits, signed=signed, symmetric=symmetric_weight, qmode='per_tensor')

    def freeze(self, qi=None, qo=None):
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self,'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # numerator
        self.mean.freeze(qi=self.qi)
        self.sub.freeze(qi1=self.qi, qi2=self.mean.qo)
        
        # denominator
        self.var_mul.freeze(qi1=self.sub.qo, qi2=self.sub.qo)
        self.var_mean.freeze(qi=self.var_mul.qo)
        self.var_sqrt.freeze(qi=self.var_mean.qo)

        # numerator / denominator
        self.div.freeze(qi1=self.sub.qo, qi2=self.var_sqrt.qo)

        if self.layernorm_module.elementwise_affine:
            self.M = self.div.qo.scale * self.qw.scale / self.qo.scale

            self.layernorm_module.weight.data = self.qw.quantize_tensor(self.layernorm_module.weight.data)
            self.layernorm_module.weight.data = self.layernorm_module.weight.data - self.qw.zero_point    # 这样减法后可能无法保证范围在 8bit 内

            self.layernorm_module.bias.data = quantize_tensor(self.layernorm_module.bias.data, scale=self.div.qo.scale * self.qw.scale,
                                                        zero_point=0, num_bits=32, signed=True)
        else:
            self.qo = self.div.qo

    def forward(self, x):
        """
        here need to get before_layer.qo as layernorm.qi
        """
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        # numerator
        mean_ = self.mean(x)
        sub_ = self.sub(x, mean_)

        # denominator
        var_mul_ = self.var_mul(sub_, sub_)
        var_mean_ = self.var_mean(var_mul_)
        var_sqrt_ = self.var_sqrt(var_mean_, qi=self.var_mean.qo)

        # numerator / denominator
        div_ = self.div(sub_, var_sqrt_, qi1=self.sub.qo, qi2=self.var_sqrt.qo)
        x = div_

        if self.layernorm_module.elementwise_affine:
            self.qw.update(self.layernorm_module.weight.data)    # 统计min、max并计算scale和zero_point
            x = torch.mul(x, FakeQuantize.apply(self.layernorm_module.weight, self.qw)) + self.layernorm_module.bias

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
            
        return x
      
    def quantize_inference(self, x, mode=None):
        x = self.qi.dequantize_tensor(x)    # float32 -> int8

        # numerator
        qmean = self.mean.quantize_inference(x, mode=mode)
        qsub = self.sub.quantize_inference(x, qmean, mode=mode)

        # denominator
        qvar_mul = self.var_mul.quantize_inference(qsub, qsub, mode=mode)
        qvar_mean = self.var_mean.quantize_inference(qvar_mul, mode=mode)
        qvar_sqrt = self.var_sqrt.quantize_inference(qvar_mean, mode=mode)

        # numerator / denominator
        qdiv = self.div.quantize_inference(qsub, qvar_sqrt, mode=mode)
        x = qdiv

        if self.layernorm_module.elementwise_affine:
            x = x - self.div.qo.zero_point

            x = x * self.layernorm_module.weight.data + self.layernorm_module.bias.data

            if mode is None:
                x = self.M * x
                x.round_() 
            elif mode == 'cmsis_nn':
                multiplier, shift = approximate_float(self.M)
                round_ = 1 << (shift - 1)
                x = (x * multiplier + round_) >> (31 - shift)
            else:
                raise Exception(f'Unknown mode {mode}')
            x = x + self.qo.zero_point        
            x.clamp_(self.qo.qmin, self.qo.qmax).round_()
        return x
    '''