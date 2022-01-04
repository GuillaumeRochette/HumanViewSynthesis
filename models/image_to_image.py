from typing import Union
from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential

from nn.blur import Blur2d
from nn.moment_normalization import SecondMomentNorm
from nn.modulated_convolution import ModulatedConv2d, ModulatedConvTranspose2d
from nn.scaled_leaky_relu import ScaledLeakyReLU

from utils import min_max


class ModulatedConvolutionLayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
        demodulate: bool = True,
        blur: bool = True,
        activation: bool = True,
    ):
        super(ModulatedConvolutionLayer, self).__init__()

        if blur and stride != 1:
            self.b = Blur2d(kernel_size=5, padding=2)
        else:
            self.b = None

        self.n = SecondMomentNorm(num_channels=in_channels)

        self.c = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            demodulate=demodulate,
        )

        if activation:
            self.a = ScaledLeakyReLU(inplace=True)
        else:
            self.a = None

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        if self.b is not None:
            x = self.b(x)
        x = self.n(x)
        x = self.c(x, s)
        if self.a is not None:
            x = self.a(x)
        return x


class ModulatedTransposedConvolutionLayer(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        output_padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
        demodulate: bool = True,
        blur: bool = True,
        activation: bool = True,
    ):
        super(ModulatedTransposedConvolutionLayer, self).__init__()

        self.n = SecondMomentNorm(num_channels=in_channels)

        self.c = ModulatedConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias,
            demodulate=demodulate,
        )

        if blur and stride != 1:
            self.b = Blur2d(kernel_size=5, padding=2)
        else:
            self.b = None

        if activation:
            self.a = ScaledLeakyReLU(inplace=True)
        else:
            self.a = None

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        x = self.n(x)
        x = self.c(x, s)
        if self.b is not None:
            x = self.b(x)
        if self.a is not None:
            x = self.a(x)
        return x


class StyledSequential(Sequential):
    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        for layer in self:
            x = layer(x, s)
        return x


class InBlock(StyledSequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedConvolutionLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        layer = ModulatedConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        super(InBlock, self).__init__(*layers)


class DownBlock(StyledSequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedConvolutionLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            style_channels=style_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        layers += [layer]
        layer = ModulatedConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        super(DownBlock, self).__init__(*layers)


class MidBlock(StyledSequential):
    def __init__(
        self,
        channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedConvolutionLayer(
            in_channels=channels,
            out_channels=channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        layer = ModulatedConvolutionLayer(
            in_channels=channels,
            out_channels=channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        super(MidBlock, self).__init__(*layers)


class UpBlock(StyledSequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedTransposedConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        layers += [layer]
        super(UpBlock, self).__init__(*layers)


class MergeBlock(StyledSequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        super(MergeBlock, self).__init__(*layers)


class OuterBlock(StyledSequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedTransposedConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        layers += [layer]
        layer = ModulatedConvolutionLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=3,
            padding=1,
        )
        layers += [layer]
        super(OuterBlock, self).__init__(*layers)


class OutBlock(StyledSequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
    ):
        layers = []
        layer = ModulatedConvolutionLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=1,
            demodulate=False,
            activation=False,
        )
        layers += [layer]
        super(OutBlock, self).__init__(*layers)


class ImageToImage(Module):
    def __init__(
        self,
        inner_blocks: int,
        mid_blocks: int,
        outer_blocks: int,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        base_channels: int,
    ):
        super(ImageToImage, self).__init__()

        self.in_block = InBlock(
            in_channels=in_channels,
            out_channels=base_channels,
            style_channels=style_channels,
        )

        e_in = e_out = 0

        layers = []
        for i in range(inner_blocks):
            e_in = min_max(i, m=0, M=3)
            e_out = min_max(i + 1, m=0, M=3)
            layer = DownBlock(
                in_channels=base_channels * 2 ** e_in,
                out_channels=base_channels * 2 ** e_out,
                style_channels=style_channels,
            )
            layers += [layer]
        self.down_blocks = ModuleList(layers)

        layers = []
        for i in range(mid_blocks):
            layer = MidBlock(
                channels=base_channels * 2 ** e_out,
                style_channels=style_channels,
            )
            layers += [layer]
        self.mid_blocks = ModuleList(layers)

        layers = []
        for i in range(inner_blocks):
            e_in = min_max(inner_blocks - i, m=0, M=3)
            e_out = min_max(inner_blocks - i - 1, m=0, M=3)
            layer = UpBlock(
                in_channels=base_channels * 2 ** e_in,
                out_channels=base_channels * 2 ** e_out,
                style_channels=style_channels,
            )
            layers += [layer]
        self.up_blocks = ModuleList(layers)

        layers = []
        for i in range(inner_blocks):
            e_in = min_max(inner_blocks - i, m=0, M=3 + 1)
            e_out = min_max(inner_blocks - i - 1, m=0, M=3)
            layer = MergeBlock(
                in_channels=base_channels * 2 ** e_in,
                out_channels=base_channels * 2 ** e_out,
                style_channels=style_channels,
            )
            layers += [layer]
        self.merge_blocks = ModuleList(layers)

        layers = []
        for i in range(outer_blocks):
            e_in = min_max(i, m=0, M=3)
            e_out = min_max(i + 1, m=0, M=3)
            layer = OuterBlock(
                in_channels=base_channels // 2 ** e_in,
                out_channels=base_channels // 2 ** e_out,
                style_channels=style_channels,
            )
            layers += [layer]
        self.outer_blocks = ModuleList(layers)

        self.out_block = OutBlock(
            in_channels=base_channels // 2 ** e_out,
            out_channels=out_channels,
            style_channels=style_channels,
        )

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        z = []

        x = self.in_block(x, s)
        z += [x]

        for i, down in enumerate(self.down_blocks):
            x = down(x, s)
            z += [x]

        for i, mid in enumerate(self.mid_blocks):
            x = (x + mid(x, s)) / sqrt(2.0)

        for i, (up, merge) in enumerate(zip(self.up_blocks, self.merge_blocks)):
            x = up(x, s)
            y = z[(len(z) - 1) - (i + 1)]
            x = torch.cat([x, y], dim=-3)
            x = merge(x, s)

        for i, outer in enumerate(self.outer_blocks):
            x = outer(x, s)

        x = self.out_block(x, s)

        x = 0.5 * (x + 1.0)

        return x


if __name__ == "__main__":
    m = ImageToImage(
        inner_blocks=4,
        mid_blocks=2,
        outer_blocks=1,
        in_channels=16,
        out_channels=3,
        style_channels=117 * 16,
        base_channels=32,
    )
    print(m)
    x = torch.randn(1, 16, 256, 256)
    s = torch.randn(1, 117 * 16)
    y = m(x, s)
    print(y.shape)
