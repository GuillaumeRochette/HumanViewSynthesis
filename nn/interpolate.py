from typing import Union

from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class Interpolate(Module):
    def __init__(
        self,
        size: Union[int, tuple] = None,
        scale_factor: Union[float, tuple] = None,
        mode: str = "nearest",
        align_corners: bool = None,
        recompute_scale_factor: bool = False,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(
            input=x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )


class Downsample(Interpolate):
    def __init__(
        self,
        scale_factor: Union[float, tuple],
        mode: str = "bilinear",
    ):
        assert scale_factor <= 1.0
        super(Downsample, self).__init__(scale_factor=scale_factor, mode=mode)


class Upsample(Interpolate):
    def __init__(
        self,
        scale_factor: Union[float, tuple],
        mode: str = "bilinear",
    ):
        assert scale_factor >= 1.0
        super(Upsample, self).__init__(scale_factor=scale_factor, mode=mode)
