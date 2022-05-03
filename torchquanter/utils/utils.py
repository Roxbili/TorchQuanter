import torch
import torch.nn as nn

def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    """
    calculate scale and zero point for quantization
    """
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    # HACK 这里scale最好和硬件保持一致，使用2的幂次，即可以用移位实现(参考一下CMSIS-NN)
    scale = (max_val - min_val) / (qmax - qmin) # S=(rmax-rmin)/(qmax-qmin)

    zero_point = qmax - max_val / scale    # Z=round(qmax-rmax/scale)

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point

def quantize_tensor(x: torch.Tensor, scale, zero_point, num_bits=8, signed=False):
    """
    use scale and zero_point to quantize tensor
    """
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2.**num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()     # q=round(clip(r/S+Z))
    
    return q_x.float()  # 由于pytorch不支持int类型的运算，因此我们还是用float来表示整数

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)    # r=S(q-Z)