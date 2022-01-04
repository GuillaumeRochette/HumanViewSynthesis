from typing import Union

from math import sqrt

from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn import init
from torch.nn import functional as F


class ScaledConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super(ScaledConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        self.scale = 1.0 / sqrt(fan_in)

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.scale * self.weight
        return self._conv_forward(input, weight, self.bias)


class ScaledConvTranspose2d(ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        output_padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super(ScaledConvTranspose2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        _, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        self.scale = 1.0 / sqrt(fan_out)

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        output_padding = self._output_padding(
            input,
            None,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        weight = self.scale * self.weight
        return F.conv_transpose2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
