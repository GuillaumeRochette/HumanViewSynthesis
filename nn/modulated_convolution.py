from typing import Union, Optional

from math import sqrt

import torch
from torch.types import _int, _size
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn import init
from torch.nn import functional as F

from nn.scaled_linear import ScaledLinear


def modulated_conv2d(
    input: Tensor,
    weight: Tensor,
    modulation: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: Union[_int, _size] = 0,
    dilation: Union[_int, _size] = 1,
    demodulate: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    ni, ci, hi, wi = input.shape
    ow, iw, hw, ww = weight.shape
    nm, cm = modulation.shape

    assert ci == iw == cm
    assert ni == nm

    modulation = modulation.reshape(nm, 1, cm, 1, 1)
    weight = weight.reshape(1, ow, iw, hw, ww)

    weight = weight * modulation

    if demodulate:
        demodulation = weight.square().sum(dim=[-3, -2, -1], keepdim=True).sqrt()
        weight = weight / (demodulation + eps)

    input = input.reshape(1, ni * ci, hi, wi)
    weight = weight.reshape(ni * ow, iw, hw, ww)

    if bias is not None:
        (ob,) = bias.shape
        assert ow == ob
        bias = bias.expand(ni, ob).reshape(ni * ob)

    output = F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=ni,
    )
    no, co, ho, wo = output.shape
    output = output.reshape(ni, ow, ho, wo)

    return output


def modulated_conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    modulation: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[_int, _size] = 1,
    padding: Union[_int, _size] = 0,
    output_padding: Union[_int, _size] = 0,
    dilation: Union[_int, _size] = 1,
    demodulate: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    ni, ci, hi, wi = input.shape
    iw, ow, hw, ww = weight.shape
    nm, cm = modulation.shape

    assert ci == iw == cm
    assert ni == nm

    modulation = modulation.reshape(nm, cm, 1, 1, 1)
    weight = weight.reshape(1, iw, ow, hw, ww)

    weight = weight * modulation

    if demodulate:
        demodulation = weight.square().sum(dim=[-4, -2, -1], keepdim=True).sqrt()
        weight = weight / (demodulation + eps)

    input = input.reshape(1, ni * ci, hi, wi)
    weight = weight.reshape(ni * iw, ow, hw, ww)

    if bias is not None:
        (ob,) = bias.shape
        assert ow == ob
        bias = bias.expand(ni, ob).reshape(ni * ob)

    output = F.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=ni,
        dilation=dilation,
    )
    no, co, ho, wo = output.shape
    output = output.reshape(ni, ow, ho, wo)

    return output


class ModulatedConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        demodulate: bool = True,
        eps: float = 1e-8,
    ):
        super(ModulatedConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_channels = style_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = Parameter(
            Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.demodulate = demodulate
        self.eps = eps

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        self.scale = 1.0 / sqrt(fan_in)

        self.linear = ScaledLinear(
            in_features=style_channels,
            out_features=in_channels,
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        init.ones_(self.linear.bias)

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        modulation = self.linear(style)
        weight = self.scale * self.weight
        bias = self.bias

        return modulated_conv2d(
            input=input,
            weight=weight,
            modulation=modulation,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            demodulate=self.demodulate,
            eps=self.eps,
        )

    def extra_repr(self):
        s = (
            "{in_channels}"
            ", {out_channels}"
            ", kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != 0:
            s += ", padding={padding}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class ModulatedConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        demodulate: bool = True,
        eps: float = 1e-8,
    ):
        super(ModulatedConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_channels = style_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

        self.weight = Parameter(
            Tensor(in_channels, out_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.demodulate = demodulate
        self.eps = eps

        _, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        self.scale = 1.0 / sqrt(fan_out)

        self.linear = ScaledLinear(
            in_features=style_channels,
            out_features=in_channels,
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        init.ones_(self.linear.bias)

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        modulation = self.linear(style)
        weight = self.scale * self.weight
        bias = self.bias

        return modulated_conv_transpose2d(
            input=input,
            weight=weight,
            modulation=modulation,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            demodulate=self.demodulate,
            eps=self.eps,
        )

    def extra_repr(self):
        s = (
            "{in_channels}"
            ", {out_channels}"
            ", kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != 0:
            s += ", padding={padding}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.output_padding != 0:
            s += ", output_padding={output_padding}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


if __name__ == "__main__":
    c = ModulatedConv2d(
        in_channels=64,
        out_channels=32,
        style_channels=128,
        kernel_size=3,
        stride=2,
        padding=1,
        demodulate=True,
    )

    # c = ModulatedConvTranspose2d(
    #     in_channels=64,
    #     out_channels=32,
    #     style_channels=128,
    #     kernel_size=3,
    #     stride=2,
    #     padding=1,
    #     output_padding=1,
    #     demodulate=True,
    # )

    x = torch.randn(1, 64, 256, 256)
    s = torch.randn(1, 128)
    y = c(x, s)
    print(y.shape, y.mean(), y.std())
