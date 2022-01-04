from torch import Tensor
from torch.nn import Module

from operations.reduce import _weighted_reduction


class WeightedMSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super(WeightedMSELoss, self).__init__()
        self.reduce = _weighted_reduction(reduction)

    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        assert not target.requires_grad
        assert not weight.requires_grad

        delta = (input - target).square()

        return self.weighted_reduce(input=delta, weight=weight)
