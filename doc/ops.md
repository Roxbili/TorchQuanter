# TorchQuanter operators

## LayerNorm

The layernrom in Pytorch
```python
import torch

x = torch.rand(32, 64)
mean_ = x.mean(dim=-1, keepdim=True)
var_ = torch.sum((x - mean_)**2, dim=-1, keepdim=True) / x.shape[-1]

output1 = (x - mean_) / torch.sqrt(var_)
output2 = torch.nn.functional.layer_norm(x, (64,))

print(output1.shape)
print(output2.shape)
print(output1[0][0:10])
print(output2[0][0:10])
```

Here is output:
```
torch.Size([32, 64])
torch.Size([32, 64])
tensor([ 0.3549, -0.2236,  0.0240, -0.1606,  0.7370,  0.9559,  1.2528,  0.6228,
         0.3453,  0.8060])
tensor([ 0.3549, -0.2236,  0.0240, -0.1605,  0.7370,  0.9558,  1.2528,  0.6228,
         0.3453,  0.8060])
```

We can find the denominator of `var_` is not `n-1` but `n`, or it will not calculate the same result with `layer_norm`.

The explanation may to use `torch.var(x, unbiased=False)` unbiased when calculating variance. Details can be seen in [link](https://stackoverflow.com/questions/66289517/layer-normalization-in-pytorch)

### Quantization inference

<!-- 
推理由两部分组成，即
```
x -> QNorm -> * W + bias -> y
```

QNorm是标准化，理论上标准化是输出fp的数值，但可以通过缩放使其放大到int32范围，
那么数学上等价之后就要除以$2^{31}$，那就可以把这个除以$2^{31}$给融合到output_scale里面，因此如果不做requantize，norm输出是一个int32的整型了。

QNorm推理过程如下：

```python
def quantize_inference(self, x, mode=None):
    x = x - self.qi.zero_point  # x 是int8的输入

    # Interger-only LayerNorm
    mean_ = x.mean(dim=-1, keepdim=True)    # int16
    sum_ = torch.sum((x - mean_)**2, dim=-1, keepdim=True).clamp(*get_qmin_qmax(self.max_bits, signed=True))    # 裁剪到32bit范围内
    var_ = torch.floor(sum_ / x.shape[-1])
    var_[var_ == 0.] = 1.   # prevent overflow
    # std_ = sqrt_interger(var_)  # 比较费时间，此处快速评估无需使用
    std_ = torch.sqrt(var_).floor()
    factor = torch.floor(2**(self.max_bits - 1) / std_)
    x = torch.floor((x - mean_) * factor / 2)

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
```

下一步 * W + bias 推理过程如下：
```python
def quantize_inference(self, x, mode=None):
    x = self.qnorm.quantize_inference(x, mode=mode)

    if self.layernorm_module.elementwise_affine:
        x = x - self.qnorm.qo.zero_point

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
```

完整的代码实现可参考：
1. [QLayerNorm](../torchquanter/nn/qlayernorm.py)
2. [QNorm](../torchquanter/nn/qnorm.py)
-->

推理由 标准化 * W + bias 组成，由于标准化中存在除法会导致输出为小数，
因此会对其进行放大，即乘上$2^{8-1}$次方，那么数学上等价之后就要除以$2^{8-1}$，
那就可以把这个除以$2^{8-1}$给融合到output_scale里面

QLayerNorm量化推理
```python
def quantize_inference(self, x, mode=None):
    x = x - self.qi.zero_point

    # Interger-only LayerNorm
    mean_ = x.mean(dim=-1, keepdim=True)    # int16
    sum_ = torch.sum((x - mean_)**2, dim=-1, keepdim=True).clamp(*get_qmin_qmax(self.max_bits, signed=True))    # 裁剪到32bit范围内
    var_ = torch.floor(sum_ / x.shape[-1])
    var_[var_ == 0.] = 1.   # prevent overflow
    # std_ = sqrt_interger(var_)  # 比较费时间，此处快速评估无需使用
    std_ = torch.sqrt(var_).floor()
    factor = torch.floor(2**(8 - 1) / std_)
    x = torch.floor(torch.clamp((x - mean_) * factor, *get_qmin_qmax(16, signed=True)))

    if self.layernorm_module.elementwise_affine:
        x = x * self.layernorm_module.weight.data + self.layernorm_module.bias.data
        x = x.clamp(*get_qmin_qmax(self.max_bits, signed=True))

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
```

完整的代码实现可参考：
1. [QLayerNorm](../torchquanter/nn/qlayernorm.py)

---------------------------------


## Softmax
The input and output of Softmax is `int8` or `uint8`. 
But it will calculate with floating point in PC, and it will calculate with fixed point in Micro.

input zero_point for softmax is useless in softmax.

**output scale is fixed to** `1/256`**, zero_point is fixed to** `-128`.   
[link1](https://stackoverflow.com/questions/54052091/softmax-tensorflow-lite-not-behaving-properly/54584333#54584333)  

### Softmax in CMSIS-NN
Note that the `arm_softmax_s8` in CMSIS-NN needs parameters `mult` and `shift`, which is different from other layers.

other layer:
```python
approximate_float(input_scale)
```

softmax in CMSIS-NN `scale` generate:
```python
softmax_input_integer_bits = 5  # 8bit定点数中整型占5bit，应该不用修改

# 这里将softmax输入的scale重新生成为接口需要的
input_scale = min(input_scale * (1 << (31 - softmax_input_integer_bits)),
                                    (1 << 31) - 1)
# 使用函数得到 arm_softmax_s8 所需要的 mult 和 shift
approximate_float(input_scale)
```
[reference link](https://github.com/ARM-software/CMSIS_5/blob/cf675280148688a50834e7b0496022360e5431cd/CMSIS/NN/Tests/UnitTest/generate_test_data.py#L781)


example:
```python
def test_softmax_s8():
    # 数据来自官方测试用例
    input_data = torch.tensor([-80, -48, 16, 0, -96], dtype=torch.float32)
    gold_output = torch.tensor([-128, -125, 56, -60, -128], dtype=torch.float32)
    input_mult = 1077952576
    input_left_shift = 23
    diff_min = -248 # 暂时不知道干什么用的

    # softmax 不需要input_zero_point，数学上不影响结果
    x = input_data - input_data.max()

    # 这里应该是官方计算中从 int8 -> fixed point 的方法
    x = ((x * input_mult) >> (31 - input_left_shift)) / (1 << (31 - 5))

    # 转成 fixed point后直接输入softmax函数中进行测试，结果正确
    out1 = F.softmax(x, dim=-1)
    out1 = out1 / (1 / 256.) - 128  # output scale和zero_point是定死的
    out1.round_()
    assert (out1 == gold_output).all(), print(out1)
```