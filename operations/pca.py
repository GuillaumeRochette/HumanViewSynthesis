from typing import Tuple
import torch
from torch import Tensor


def pca(x: Tensor) -> Tuple[Tensor, ...]:
    mu = x.mean(dim=0, keepdim=True)
    U, S, V = (x - mu).t().svd()
    return U, S, V


class Features2Image(object):
    def __init__(self, num_channels: int):
        self.num_channels = num_channels
        self.U = None

    def __call__(self, x: Tensor) -> Tensor:
        assert 3 <= x.dim() <= 4
        if x.dim() == 3:
            x.unsqueeze(0)

        s = x.shape[2:]
        x = x.flatten(start_dim=2)
        x = x.transpose(-1, -2)

        y = []
        for e in x:
            if self.U is None:
                self.U, _, _ = pca(e)
            y.append(e @ self.U[:, : self.num_channels])
        y = torch.stack(y)

        y = y.transpose(-1, -2)
        y = y.reshape(y.shape[:2] + s)
        return y
