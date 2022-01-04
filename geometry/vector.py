from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def homogeneous(x: Tensor) -> Tensor:
    """
    Converts heterogeneous vector into homogeneous vector.

    :param x: Heterogeneous vector of shape [*, N, 1].
    :return: Homogeneous vector of shape [*, N + 1, 1].
    """
    assert x.shape[-1] == 1

    s = x.shape[:-2] + (1, 1)
    o = torch.ones(s, dtype=x.dtype, device=x.device)
    y = torch.cat([x, o], dim=-2)

    return y


def heterogeneous(x: Tensor) -> Tensor:
    """
    Converts homogeneous vector into heterogeneous vector.

    :param x: Homogeneous vector of shape [*, N + 1, 1].
    :return: Heterogeneous vector of shape [*, N, 1].
    """
    assert x.shape[-1] == 1

    n = x.shape[-2] - 1
    a, b = x.split([n, 1], dim=-2)
    x = a / b

    return x


def norm(x: Tensor, p: int = 2) -> Tensor:
    """
    Returns the norm of a vector.

    :param x: Vector of shape [*, N, 1].
    :param p: Order of the norm.
    :return: Norm of the vector of shape [*, 1, 1].
    """
    assert x.shape[-1] == 1

    return x.norm(p=p, dim=-2, keepdim=True)


def pnorm(x: Tensor, p: int = 2) -> Tensor:
    """
    Returns the p-norm of a vector.

    :param x: Vector of shape [*, N, 1].
    :param p: Order of the norm.
    :return: P-Norm of the vector of shape [*, 1, 1].
    """
    assert x.shape[-1] == 1

    return x.pow(exponent=p).sum(dim=-2, keepdim=True)


def normalize(x: Tensor, p: int = 2, eps: float = 1e-8) -> Tensor:
    """
    Returns a vector with a unit p-norm.

    :param x: Vector of shape [*, N, 1].
    :param p: Order of the norm.
    :param eps: Stability parameter to avoid division by 0.
    :return: Vector of shape [*, N, 1].
    """
    assert x.shape[-1] == 1

    return F.normalize(x, p=p, dim=-2, eps=eps)


def cosine_similarity(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute the cosine similarity between two vectors.

    :param x: Vector of shape [*, N, 1].
    :param y: Vector of shape [*, N, 1].
    :param eps: Stability parameter to avoid division by 0.
    :return: Vector of shape [*, 1, 1].
    """
    assert x.shape[-2] == y.shape[-2]
    assert x.shape[-1] == y.shape[-1] == 1

    return F.cosine_similarity(x, y, dim=-2, eps=eps).unsqueeze(dim=-1)


def inner(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the inner product  (or dot product) of two vectors.

    :param x: Vector of shape [*, N, 1].
    :param y: Vector of shape [*, N, 1].
    :return: Inner product of the vectors of shape [*, 1, 1].
    """
    assert x.shape[-2] == y.shape[-2]
    assert x.shape[-1] == y.shape[-1] == 1

    return x.transpose(-1, -2) @ y


def outer(x: Tensor, y: Tensor) -> Tensor:
    """
    Returns the inner product  (or dot product) of two vectors.

    :param x: Vector of shape [*, N, 1].
    :param y: Vector of shape [*, N, 1].
    :return: Outer product of the vectors of shape [*, N, N].
    """
    assert x.shape[-2] == y.shape[-2]
    assert x.shape[-1] == y.shape[-1] == 1

    return x @ y.transpose(-1, -2)


def split(a: Tensor, index: int, dim: int) -> Tuple[Tensor, Tensor]:
    """
    Splits a tensor at a given index and returns the split part and the rest.

    :param a: Tensor.
    :param index: Index of split.
    :param dim: Dimension of split.
    :return:
        - Split tensor.
        - Rest tensor.
    """
    n = a.shape[dim]
    s, b, e = a.split([index, 1, n - (index + 1)], dim=dim)
    c = torch.cat([s, e], dim=dim)
    return b, c


def reunite(a: Tensor, b: Tensor, index: int, dim: int) -> Tensor:
    """
    Reunites by inserting the first tensor at a given index and dimension in the second tensor.

    :param a: First tensor.
    :param b: Second tensor.
    :param index: Index of insertion.
    :param dim: Dimension of insertion.
    :return: Reunited tensor.
    """
    m = b.shape[dim]
    s, e = b.split([index, m - index], dim=dim)
    c = torch.cat([s, a, e], dim=dim)
    return c
