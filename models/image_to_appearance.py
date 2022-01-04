from typing import Tuple

import torch
from torch import Tensor, Size
from torch.nn import Module, ModuleList
from torch.nn import Linear
from torch.nn import Flatten, Unflatten

from nn.resnet_gn import resnet50_gn


class ImageToAppearance(Module):
    def __init__(
        self,
        out_dims: Tuple[int, ...],
        pretrained: bool = True,
    ):
        super(ImageToAppearance, self).__init__()
        out_dims = Size(out_dims)

        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)

        std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        self.register_buffer("std", std, persistent=False)

        m = resnet50_gn(pretrained=pretrained)
        layers = [
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool,
            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
            m.avgpool,
            Flatten(start_dim=-3),
            Linear(
                in_features=2048,
                out_features=out_dims.numel(),
            ),
            Unflatten(dim=-1, unflattened_size=out_dims),
        ]
        self.layers = ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    m = ImageToAppearance(
        out_dims=(117, 16),
        pretrained=True,
    )
    print(m)
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print(y.shape)
