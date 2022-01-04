from math import sqrt

from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class ScaledLeakyReLU(Module):
    def __init__(
        self,
        negative_slope: float = 0.2,
        scale: float = sqrt(2.0),
        inplace: bool = False,
    ):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.scale = scale

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope, self.inplace) * self.scale

    def extra_repr(self):
        s = (
            f"negative_slope={self.negative_slope}"
            f", scale={round(self.scale, 4)}"
            ", inplace=True"
            if self.inplace
            else ""
        )
        return s
