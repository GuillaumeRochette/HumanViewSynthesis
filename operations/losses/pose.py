from torch import Tensor
from torch.nn import Module

from geometry import vector
from operations.reduce import _reduction, _weighted_reduction


class MPJPELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super(MPJPELoss, self).__init__()
        self.reduce = _reduction(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert not target.requires_grad

        delta = vector.norm(input - target)

        return self.reduce(input=delta)


class MPVJPELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super(MPVJPELoss, self).__init__()
        self.reduce = _weighted_reduction(reduction)

    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        assert not target.requires_grad
        assert not weight.requires_grad

        delta = vector.norm(input - target)

        return self.reduce(input=delta, weight=weight)
