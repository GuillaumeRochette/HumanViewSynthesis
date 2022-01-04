from typing import Union, Iterable

import torch
from torch import Tensor, dtype


def _reduction(reduction: str):
    if reduction == "sum":
        return torch.sum
    elif reduction == "mean":
        return torch.mean
    elif reduction == "none":
        return lambda input: input
    else:
        raise ValueError(f"Invalid reduction: {reduction}.")


def none(input: Tensor) -> Tensor:
    return input


def _weighted_reduction(reduction: str):
    if reduction == "sum":
        return weighted_sum
    elif reduction == "mean":
        return weighted_mean
    elif reduction == "none":
        return weighted_none
    else:
        raise ValueError(f"Invalid reduction: {reduction}.")


def weighted_sum(
    input: Tensor,
    weight: Tensor,
    dim: Union[int, Iterable[int]] = (),
    keepdim: bool = False,
    dtype: dtype = None,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype

    return (input * weight).sum(dim=dim, keepdim=keepdim, dtype=dtype)


def weighted_mean(
    input: Tensor,
    weight: Tensor,
    dim: Union[int, Iterable[int]] = (),
    keepdim: bool = False,
    dtype: dtype = None,
    eps: float = 1e-8,
) -> Tensor:
    if dtype is None:
        dtype = input.dtype

    numerator = (input * weight).sum(dim=dim, keepdim=keepdim, dtype=dtype)
    denominator = weight.sum(dim=dim, keepdim=keepdim, dtype=dtype)
    denominator = denominator.maximum(torch.full_like(denominator, fill_value=eps))

    return numerator / denominator


def weighted_none(input: Tensor, weight: Tensor) -> Tensor:
    return input * weight
