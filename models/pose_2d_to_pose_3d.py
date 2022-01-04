from math import sqrt

import torch
from torch import Tensor, Size
from torch.nn import Module, Sequential
from torch.nn import Linear, GroupNorm, ReLU, Dropout


class LinearLayer(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super(LinearLayer, self).__init__()

        self.linear = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        self.norm = GroupNorm(num_groups=32, num_channels=out_features)
        self.activation = ReLU(inplace=True)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualLinearLayer(Module):
    def __init__(self, features: int, dropout: float):
        super(ResidualLinearLayer, self).__init__()
        self.layers = Sequential(
            LinearLayer(in_features=features, out_features=features, dropout=dropout),
            LinearLayer(in_features=features, out_features=features, dropout=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = (x + self.layers(x)) / sqrt(2.0)
        return x


def scale_by_width(p: Tensor, resolution: Tensor) -> Tensor:
    x, y = p.split(1, dim=-2)

    w, h = resolution.split(1, dim=-1)
    w, h = w[..., None, None, :], h[..., None, None, :]

    x = (2.0 * x - w) / w
    y = (2.0 * y - h) / w

    p = torch.cat([x, y], dim=-2)

    return p


class Pose2DToPose3D(Module):
    def __init__(
        self,
        n_joints: int,
        d_embedding: int,
        dropout: float,
    ):
        super(Pose2DToPose3D, self).__init__()

        self.in_dims = Size([n_joints, 2 + 1, 1])
        self.out_dims = Size([n_joints, 3, 1])

        in_features = self.in_dims.numel()
        out_features = self.out_dims.numel()

        self.layers = Sequential(
            LinearLayer(
                in_features=in_features,
                out_features=d_embedding,
                dropout=dropout,
            ),
            ResidualLinearLayer(features=d_embedding, dropout=dropout),
            ResidualLinearLayer(features=d_embedding, dropout=dropout),
            Linear(
                in_features=d_embedding,
                out_features=out_features,
            ),
        )

    def forward(self, p: Tensor, c: Tensor, resolution: Tensor) -> Tensor:
        p = scale_by_width(p=p, resolution=resolution)
        x = torch.cat([p, c], dim=-2)

        x = x.reshape(-1, self.in_dims.numel())
        x = self.layers(x)
        x = x.reshape((-1,) + self.out_dims)

        return x


if __name__ == "__main__":
    m = Pose2DToPose3D(
        n_joints=117,
        d_embedding=1024,
        dropout=0.1,
    )
    print(m)
    p = torch.randn(1, 117, 2, 1)
    c = torch.randn(1, 117, 1, 1)
    r = torch.randn(1, 2)

    y = m(p, c, r)
    print(y.shape)
