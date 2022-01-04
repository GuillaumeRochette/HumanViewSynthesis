import torch
from torch import Tensor
from torch.nn import Module


class ExponentialMovingAverage(Module):
    def __init__(self, *size: int, momentum: float = 0.995):
        super(ExponentialMovingAverage, self).__init__()

        self.register_buffer("average", torch.ones(*size))
        self.register_buffer("initialised", torch.tensor(False))
        self.momentum = momentum

    @torch.no_grad()
    def forward(self, x: Tensor):
        if self.training:
            self.update(x=x)

        return self.average

    def update(self, x: Tensor):
        if self.initialised.all():
            self.average.copy_(x.lerp(self.average, self.momentum))
        else:
            self.average.copy_(x)
            self.initialised.copy_(~self.initialised)
