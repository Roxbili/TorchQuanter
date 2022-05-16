# TorchQuanter

TorchQuanter is designed to quantize pytorch model.

## Quantization specification
```python
signed
    True:  int8
    False: uint8

symmetric_weight:
    True: int8, zero_point=0

qmode:
    per_tensor:
        * "module weights share the same scale and zero_point"

    per_channel: 
        * "each output channel of module weights has a scale and a zero_point"
        * support op: Conv, Depthwise-conv
```

## Support operations
```
Conv2d, Conv2d + BatchNorm2d + ReLU
Linear, Linear + ReLU
ReLU
MaxPool2d
```

## TODO
- [x] 量化框架Linear、ReLU融合
- [x] 量化框架uint8和int8之间可选
- [x] 是否对称量化可选
- [x] per-tensor 和 per-channel之间可选
- [x] 量化框架scale修改实现，使用multiplier和shift
- [x] 支持add
- [x] 支持layernorm
- [x] 支持softmax
- [x] 支持mul量化
- [x] 支持matmul量化
- [x] 支持attention
- [x] 支持transformer encoder