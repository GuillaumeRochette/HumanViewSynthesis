from typing import Tuple

import torch
from torch import Tensor

from geometry import matrix


def forward_kinematics(
    R: Tensor,
    l: Tensor,
    edges: Tuple[Tuple[int, int]],
) -> Tuple[Tensor, Tensor]:
    """
    Computes the forward kinematics chain, given joint angles, limb lengths and the edges of the graph.

    :param R: Rotation matrices in local coordinates frames of shape [*, J - 1, N, N].
    :param l: Limb lengths of shape [*, J - 1, 1, 1].
    :param edges: Ordered edges of the graph of length J - 1.
    :return: Rotation matrices and location vectors in the global coordinate frame of shape [*, J - 1, N, N] and [*, J - 1, N, 1].
    """
    assert R.shape[:-3] == l.shape[:-3]
    assert R.shape[-3] == l.shape[-3] == len(edges)
    assert R.shape[-2] == R.shape[-1]
    assert l.shape[-2] == l.shape[-1] == 1

    e = matrix.eye_like(R).split(1, dim=-1)[0]
    S = matrix.homogeneous(A=R, b=l * R @ e)

    S = S.split(1, dim=-3)
    T = [Tensor()] * len(edges)

    for i, j in edges:
        if i == 0:
            T[j - 1] = S[j - 1]
        else:
            T[j - 1] = T[i - 1] @ S[j - 1]

    T = torch.cat(T, dim=-3)
    R, P = matrix.heterogeneous(T)

    return R, P


def not_inverse_kinematics(
    points: Tensor,
    edges: Tuple[Tuple[int, int]],
) -> Tuple[Tensor, Tensor]:
    """

    :param points: [*, J, N, 1].
    :param edges: Ordered edges of the graph of length J - 1.
    :return:
        - midpoints: [*, J, N, 1].
        - directions: [*, J, N, 1].
    """
    points = points.split(1, dim=-3)
    midpoints, directions = [], []

    for i, j in edges:
        start, end = points[i], points[j]
        midpoints += [(start + end) / 2]
        directions += [end - start]

    midpoints = torch.cat(midpoints, dim=-3)
    directions = torch.cat(directions, dim=-3)

    return midpoints, directions
