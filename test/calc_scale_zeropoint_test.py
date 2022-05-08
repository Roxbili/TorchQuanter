import torch
from torchquanter.utils import calcScaleZeroPoint

if __name__ == '__main__':
    tensor = torch.randn(8, 3, 3, 3)

    # per_channel
    max_ = tensor.flatten(1).max(dim=-1)[0]
    min_ = tensor.flatten(1).min(dim=-1)[0]
    print(min_)
    print(max_)
    a = calcScaleZeroPoint(min_, max_, symmetric=False)
    a = calcScaleZeroPoint(min_, max_, symmetric=True)

    # per_tensor
    max_ = tensor.max()
    min_ = tensor.min()
    print(min_)
    print(max_)
    a = calcScaleZeroPoint(min_, max_, symmetric=False)
    a = calcScaleZeroPoint(min_, max_, symmetric=True)
