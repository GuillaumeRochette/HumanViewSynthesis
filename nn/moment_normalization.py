import torch
from torch import Tensor
from torch.nn import Module

from nn.exponential_moving_average import ExponentialMovingAverage


class SecondMomentNorm(Module):
    def __init__(self, num_channels: int, momentum: float = 0.995, eps: float = 1e-8):
        super(SecondMomentNorm, self).__init__()

        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        self.ema = ExponentialMovingAverage(num_channels, momentum=momentum)

    def forward(self, input: Tensor) -> Tensor:
        n, c, h, w = input.shape

        moment = input.square().mean(dim=[-4, -2, -1])
        moment = self.ema(moment)
        moment = moment.reshape(1, c, 1, 1)

        num = input
        den = moment.sqrt()
        den = torch.maximum(den, torch.full_like(den, fill_value=self.eps))
        output = num / den

        return output

    def extra_repr(self):
        s = f"num_channels={self.num_channels}" f", momentum={self.momentum}"
        return s
