from typing import Tuple
import torch
from torch import Tensor


def homogeneous(A: Tensor, b: Tensor) -> Tensor:
    """
    Converts heterogeneous matrix into homogeneous matrix.

    :param A: Heterogeneous matrix of shape [*, N, N].
    :param b: Heterogeneous vector of shape [*, N, 1].
    :return: Homogeneous matrix of shape [*, N + 1, N + 1].
    """
    assert A.shape[:-2] == b.shape[:-2]
    assert A.shape[-2] == A.shape[-1] == b.shape[-2]
    assert b.shape[-1] == 1

    s, n = A.shape[:-2], A.shape[-2]

    c = torch.zeros(s + (1, n), dtype=A.dtype, device=A.device)
    d = torch.ones(s + (1, 1), dtype=A.dtype, device=A.device)

    M = torch.cat(
        [
            torch.cat([A, b], dim=-1),
            torch.cat([c, d], dim=-1),
        ],
        dim=-2,
    )

    return M


def heterogeneous(M: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Converts homogeneous matrix into heterogeneous matrix.

    :param M: Homogeneous matrix of shape [*, N + 1, N + 1].
    :return: Heterogeneous matrix and vector of shapes [*, N, N] and [*, N, 1] respectively.
    """
    assert M.shape[-2] == M.shape[-1]

    n = M.shape[-2] - 1

    Ab, cd = M.split([n, 1], dim=-2)
    A, b = Ab.split([n, 1], dim=-1)
    c, d = cd.split([n, 1], dim=-1)
    A, b = A / d, b / d

    return A, b


def affine(x: Tensor, A: Tensor, b: Tensor) -> Tensor:
    """
    Applies an affine transformation to x given A and b.

    :param x: Vector of shape [*, N, 1].
    :param A: Matrix of shape [*, N, N].
    :param b: Vector of shape [*, N, 1].
    :return: Vector of shape [*, N, 1].
    """
    assert x.ndim == A.ndim == b.ndim
    assert x.shape[-2] == A.shape[-2] == A.shape[-1] == b.shape[-2]
    assert x.shape[-1] == b.shape[-1] == 1

    y = A @ x + b

    return y


def eye_like(x: Tensor) -> Tensor:
    """
    Return an identity matrix of the same shape as x.

    :param x: Matrix of shape [*, M, N].
    :return: Identity matrix of shape [*, M, N].
    """
    m, n = x.shape[-2], x.shape[-1]

    return torch.eye(m, n, dtype=x.dtype, device=x.device).expand_as(x)


def diag(x: Tensor):
    """
    Returns a diagonal matrix given a vector.

    :param x: Vector of shape [*, M, 1].
    :return: Diagonal matrix of shape [*, M, M].
    """
    assert x.shape[-1] == 1
    m = x.shape[-2]

    return torch.eye(m, dtype=x.dtype, device=x.device) * x
