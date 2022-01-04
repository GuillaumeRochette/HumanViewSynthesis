from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

from geometry.kinematic import not_inverse_kinematics
from geometry.rotation import vector_rotation


class Pose3DToJoints(Module):
    def __init__(self, radius: float = 0.02):
        super(Pose3DToJoints, self).__init__()
        assert radius > 0.0

        self.radius = radius
        self.register_buffer("identity", torch.eye(3), persistent=False)
        self.register_buffer("ones", torch.ones(3, 1), persistent=False)

    def forward(self, points: Tensor) -> Tuple[Tensor, ...]:
        """

        :param points: [*, N, 3, 1]
        :return:
            - mu: [*, N, 3, 1]
            - rho: [*, N, 3, 3]
            - lambd: [*, N, 3, 1]
        """
        identity = self.identity.expand(points.shape[:-2] + self.identity.shape)
        ones = self.ones.expand(points.shape[:-2] + self.sphere.shape)

        mu = points
        rho = identity
        lambd = self.radius ** 2 * ones
        return mu, rho, lambd


class Pose3DToLimbs(Module):
    def __init__(
        self,
        edges: Tuple[Tuple[int, int]],
        widths: Tensor,
        min_length: float = 1e-3,
    ):
        super(Pose3DToLimbs, self).__init__()
        assert min_length > 0.0

        self.edges = edges
        self.min_length = min_length

        e = torch.eye(3).split(1, dim=-1)[0]
        self.register_buffer("e", e, persistent=False)
        self.register_buffer("widths", widths, persistent=False)

    def forward(self, points: Tensor) -> Tuple[Tensor, ...]:
        """

        :param points: [*, N, 3, 1]
        :return:
            - mu: [*, M, 3, 1]
            - rho: [*, M, 3, 3]
            - lambd: [*, M, 3, 1]
        """
        mu, directions = not_inverse_kinematics(points=points, edges=self.edges)
        rho = vector_rotation(x=self.e.expand_as(directions), y=directions)
        squared_lengths = directions.square().sum(dim=-2, keepdim=True)
        squared_widths = self.widths.square().expand_as(squared_lengths)
        lambd = torch.cat(
            [
                squared_lengths.clamp(min=self.min_length ** 2),
                squared_widths,
                squared_widths,
            ],
            dim=-2,
        )
        return mu, rho, lambd


class Pose3DToJointsAndLimbs(Module):
    def __init__(
        self,
        edges: Tuple[Tuple[int, int]],
        widths: Tensor,
        radius: float = 0.02,
        min_length: float = 1e-3,
    ):
        super(Pose3DToJointsAndLimbs, self).__init__()

        self.p2j = Pose3DToJoints(radius=radius)
        self.p2l = Pose3DToLimbs(edges=edges, widths=widths, min_length=min_length)

    def forward(self, points: Tensor) -> Tuple[Tensor, ...]:
        """

        :param points: [*, N, 3, 1]
        :return:
            - mu: [*, N + M, 3, 1]
            - rho: [*, N + M, 3, 3]
            - lambd: [*, N + M, 3, 1]
        """
        mu_j, (rho_j, lambd_j) = self.p2j(points=points)
        mu_l, (rho_l, lambd_l) = self.p2l(points=points)

        mu = torch.cat([mu_j, mu_l], dim=-3)
        rho = torch.cat([rho_j, rho_l], dim=-3)
        lambd = torch.cat([lambd_j, lambd_l], dim=-3)
        return mu, rho, lambd
