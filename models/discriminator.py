from typing import Union
from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

from nn.blur import Blur2d
from nn.scaled_convolution import ScaledConv2d
from nn.scaled_leaky_relu import ScaledLeakyReLU

from utils import min_max


class ScaledConvolutionLayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
        blur: bool = True,
        activation: bool = True,
    ):
        super(ScaledConvolutionLayer, self).__init__()

        self.b = None
        if blur and stride != 1:
            self.b = Blur2d(kernel_size=3, padding=1)

        self.c = ScaledConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.a = None
        if activation:
            self.a = ScaledLeakyReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.b is not None:
            x = self.b(x)
        x = self.c(x)
        if self.a is not None:
            x = self.a(x)
        return x


class ResidualScaledConvolutionLayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
        blur: bool = True,
        activation: bool = True,
    ):
        super(ResidualScaledConvolutionLayer, self).__init__()

        self.residual = Sequential(
            ScaledConvolutionLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias,
                activation=activation,
            ),
            ScaledConvolutionLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                blur=blur,
                activation=activation,
            ),
        )

        self.shortcut = ScaledConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            bias=False,
            blur=blur,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = (self.residual(x) + self.shortcut(x)) / sqrt(2.0)
        return x


class Discriminator(Module):
    def __init__(
        self,
        blocks: int = 4,
        in_channels: int = 3,
        base_channels: int = 32,
    ):
        super(Discriminator, self).__init__()

        layers = []

        layer = ScaledConvolutionLayer(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=1,
        )
        layers += [layer]

        e_in = e_out = 0
        for i in range(blocks):
            e_in = min_max(i, m=0, M=3)
            e_out = min_max(i + 1, m=0, M=3)
            layer = ResidualScaledConvolutionLayer(
                in_channels=base_channels * 2 ** e_in,
                out_channels=base_channels * 2 ** e_out,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            layers += [layer]

        layer = ScaledConvolutionLayer(
            in_channels=base_channels * 2 ** e_out,
            out_channels=1,
            kernel_size=1,
            activation=False,
        )
        layers += [layer]

        self.layers = ModuleList(layers)

        self.in_channels = in_channels
        self.out_channels = 1

    def forward(self, x: Tensor):
        y = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y += [x]
        return y[:-1], y[-1]


if __name__ == "__main__":
    m = Discriminator(blocks=4)
    print(m)
    x = torch.randn(1, 3, 256, 256)
    y, z = m(x)
    for a in y:
        print(a.shape)
    print(z.shape)
