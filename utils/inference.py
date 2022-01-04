from torch import Tensor
from torch.nn import Module


def move(d, device):
    for k in d:
        if isinstance(d[k], dict):
            d[k] = move(d[k], device=device)
        elif isinstance(d[k], (Tensor, Module)):
            d[k] = d[k].to(device=device)
    return d
