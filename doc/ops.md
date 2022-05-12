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

![img](../img/IMG_1294.png)


## Softmax
The input and output of Softmax is `int8` or `uint8`. 
But it will calculate with floating point in PC, and it will calculate with fixed point in Micro.

input zero_point for softmax is useless in softmax.

output scale is fixed to `1/256`, zero_point is fixed to `-128`.   
[link1](https://stackoverflow.com/questions/54052091/softmax-tensorflow-lite-not-behaving-properly/54584333#54584333)  

### Softmax in CMSIS-NN
Note that the `arm_softmax_s8` in CMSIS-NN needs parameters `mult` and `shift`, which is different from other layers.

other layer:
```python
approximate_float(input_scale)
```

softmax in CMSIS-NN:
```python
softmax_input_integer_bits = 5

input_scale = min(input_scale * (1 << (31 - softmax_input_integer_bits)),
                                    (1 << 31) - 1)
approximate_float(input_scale)
```
[reference link](https://github.com/ARM-software/CMSIS_5/blob/cf675280148688a50834e7b0496022360e5431cd/CMSIS/NN/Tests/UnitTest/generate_test_data.py#L781)