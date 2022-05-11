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

## Quantization inference

![img](../img/IMG_1294.png)