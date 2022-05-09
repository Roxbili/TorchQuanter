import torch
from torchquanter.utils import calcScaleZeroPoint, approximate_float

if __name__ == '__main__':
    tensor = torch.randn(8, 3, 3, 3)

    # per_channel
    max_ = tensor.flatten(1).max(dim=-1)[0]
    min_ = tensor.flatten(1).min(dim=-1)[0]
    print(min_)
    print(max_)

    scale, zero_point = calcScaleZeroPoint(min_, max_, symmetric=False)
    multiplier, shift = approximate_float(scale)
    err = abs(scale - multiplier >> (31 - shift))
    print(f'multiplier: {multiplier}, shift: {shift}, err: {err}')

    scale, zero_point = calcScaleZeroPoint(min_, max_, symmetric=True)
    multiplier, shift = approximate_float(scale)
    err = abs(scale - multiplier >> (31 - shift))
    print(f'multiplier: {multiplier}, shift: {shift}, err: {err}')


    # per_tensor
    max_ = tensor.max()
    min_ = tensor.min()
    print(min_)
    print(max_)

    scale, zero_point = calcScaleZeroPoint(min_, max_, symmetric=False)
    multiplier, shift = approximate_float(scale)
    err = abs(scale - multiplier >> (31 - shift))
    print(f'multiplier: {multiplier}, shift: {shift}, err: {err}')

    scale, zero_point = calcScaleZeroPoint(min_, max_, symmetric=True)
    multiplier, shift = approximate_float(scale)
    err = abs(scale - multiplier >> (31 - shift))
    print(f'multiplier: {multiplier}, shift: {shift}, err: {err}')