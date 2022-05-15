import torch

from torchquanter.utils import sqrt_interger

def sqrt_interger_test(tensor: torch.Tensor):
    out1 = torch.sqrt(tensor).floor()
    out2 = sqrt_interger(tensor)

    if (out1 == out2).all() == False:
        print(f'       tensor: {tensor}')
        print(f'         gold: {out1}')
        print(f'sqrt_interger: {out2}')
        raise Exception

def data_generator(tensor_size, low=0, high=2**32):
    """
    low(inclusive), high(exclusive)
    """
    return torch.randint(low=low, high=high, size=tensor_size).float()

if __name__ == '__main__':
    for i in range(1000):
        tensor = data_generator((10,))
        sqrt_interger_test(tensor)