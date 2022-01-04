from math import sqrt

from torch import Tensor
from torch.nn import Linear
from torch.nn import init
from torch.nn import functional as F


class ScaledLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ScaledLinear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
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
        return F.linear(input, weight, self.bias)
