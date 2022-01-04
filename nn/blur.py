from typing import Union
from math import floor, ceil

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


def _box_kernel_1d(n: int) -> Tensor:
    assert n > 0
    x = [1.0 for k in range(n)]
    x = torch.tensor(x)
    x = x / x.sum()
    return x


def _binomial_kernel_1d(n: int) -> Tensor:
    assert n > 0
    x = [1.0]
    for k in range(n - 1):
        y = x[k] * (n - 1 - k) / (k + 1)
        x.append(y)
    x = torch.tensor(x)
    x = x / x.sum()
    return x


def _box_kernel_2d(h: int, w: int) -> Tensor:
    a, b = _box_kernel_1d(n=h), _box_kernel_1d(n=w)
    x = a[:, None] * b[None, :]
    return x


def _binomial_kernel_2d(h: int, w: int) -> Tensor:
    a, b = _binomial_kernel_1d(n=h), _binomial_kernel_1d(n=w)
    x = a[:, None] * b[None, :]
    return x


class Blur2d(Module):
    def __init__(
        self,
        kernel_size: int,
        kernel_type: str = "binomial",
        stride: int = 1,
        padding: Union[int, float] = 0,
    ):
        super(Blur2d, self).__init__()

        self.kernel_size = kernel_size

        if kernel_type == "box":
            kernel_func = _box_kernel_2d
        elif kernel_type == "binomial":
            kernel_func = _binomial_kernel_2d
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}.")
        self.kernel_type = kernel_type

        kernel = kernel_func(h=kernel_size, w=kernel_size)
        self.register_buffer("kernel", kernel, persistent=False)

        self.stride = stride

        if isinstance(padding, float):
            p1, p2 = floor(padding), ceil(padding)
        else:
            p1 = p2 = padding

        self.padding = (p1, p2, p1, p2)

    def forward(self, input: Tensor) -> Tensor:
        n, c, h, w = input.shape
        kernel = self.kernel.expand((c, 1) + self.kernel.shape)

        return F.conv2d(
            input=F.pad(input, self.padding),
            weight=kernel,
            stride=self.stride,
            groups=c,
        )

    def extra_repr(self):
        s = (
            f"kernel_size={self.kernel_size}"
            f", kernel_type={self.kernel_type}"
            f", stride={self.stride}"
            f", padding={self.padding}"
        )
        return s
