import torch
import torch.nn as nn

def get_qmin_qmax(num_bits, signed):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2.**num_bits - 1.
    return qmin, qmax

def calcScaleZeroPoint(min_val: torch.Tensor, max_val: torch.Tensor, num_bits=8, signed=True, symmetric=False):
    """
    calculate scale and zero point for quantization
    """
    qmin, qmax = get_qmin_qmax(num_bits, signed)

    # TODO 这里scale最好和硬件保持一致，使用2的幂次，即可以用移位实现(参考一下CMSIS-NN)
    if not symmetric:
        scale = (max_val - min_val) / (qmax - qmin) # S=(rmax-rmin)/(qmax-qmin)
        zero_point = qmax - max_val / scale    # Z=round(qmax-rmax/scale)

        zero_point = torch.where(zero_point < qmin, torch.tensor(qmin, dtype=zero_point.dtype), zero_point)
        zero_point = torch.where(zero_point > qmax, torch.tensor(qmax, dtype=zero_point.dtype), zero_point)
    else:
        scale = torch.where(max_val.abs().data > min_val.abs().data, max_val.abs().data, min_val.abs().data) / max(abs(qmax), abs(qmin))
        zero_point = torch.zeros(max_val.shape, dtype=min_val.dtype)

    zero_point.round_()
    return scale.to(min_val.device), zero_point.to(min_val.device)

def quantize_tensor(x: torch.Tensor, scale, zero_point, num_bits=8, signed=True):
    """
    use scale and zero_point to quantize tensor
    """
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=x.dtype)
    if not isinstance(zero_point, torch.Tensor):
        zero_point = torch.tensor(zero_point, dtype=x.dtype)

    qmin, qmax = get_qmin_qmax(num_bits, signed)
    scale_ = broadcast_dim_as(scale, x)
    zero_point_ = broadcast_dim_as(zero_point, x)
 
    q_x = zero_point_ + x / scale_
    q_x.clamp_(qmin, qmax).round_()     # q=round(clip(r/S+Z))
    
    return q_x.float()  # 由于pytorch不支持int类型的运算，因此我们还是用float来表示整数

def dequantize_tensor(q_x, scale, zero_point):
    scale_ = broadcast_dim_as(scale, q_x)
    zero_point_ = broadcast_dim_as(zero_point, q_x)
    return scale_ * (q_x - zero_point_)    # r=S(q-Z)

def broadcast_dim_as(tensor: torch.Tensor, x: torch.Tensor, dim=0):
    """
    broadcast tensor to x's dimension.

    Args
    ----------
    tensor: the tensor to be broadcasted
    x: target tensor dimensions
    dim: which dimension to keep

    e.g.:
        x.shape (32, 1, 3, 3)
        tensor.shape (32)

        after broadcast: tensor shape is (32, 1, 1, 1)
    """
    assert tensor.dim() <= 1, 'tensor dimension must be 0 or 1'
    dims = [1 if i != dim else -1 for i in range(x.dim())]
    return tensor.view(dims)

def approximate_float(M):
    """
    approximate float with multiplier and shift.
    ```
    float = multiplier / 2^(31 - shift)
    float = multiplier >> (31 - shift)
    ```

    Args
    ----------
    M: float number

    Return
    ----------
    multiplier: torch.float32(real: int32)
    shift: torch.int32(real: int8)
    """
    significand, shift = torch.frexp(M)
    significand_q31 = torch.round(significand * (1 << 31))

    # to torch tensor
    return significand_q31, shift