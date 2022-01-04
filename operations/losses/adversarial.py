from typing import List

import torch
from torch import Tensor, autograd
from torch.nn import Module
from torch.nn import BCEWithLogitsLoss, L1Loss


class AdversarialLoss(Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.register_buffer("real", torch.tensor(1.0), persistent=False)
        self.register_buffer("fake", torch.tensor(0.0), persistent=False)
        self.bce = BCEWithLogitsLoss()

    def forward(self, input: Tensor, target: bool) -> Tensor:
        if target:
            target = self.real.expand_as(input)
        else:
            target = self.fake.expand_as(input)
        loss = self.bce(input=input, target=target)
        return loss


class AdversarialFeaturesLoss(Module):
    def __init__(self):
        super(AdversarialFeaturesLoss, self).__init__()
        self.mae = L1Loss()

    def forward(self, input: List[Tensor], target: List[Tensor]) -> Tensor:
        losses = []
        for (i, t) in zip(input, target):
            assert not t.requires_grad
            losses += [self.mae(input=i, target=t)]
        losses = torch.stack(losses)
        return losses.sum()


class R1(Module):
    def __init__(self):
        super(R1, self).__init__()

    def forward(self, outputs: Tensor, inputs: Tensor) -> Tensor:
        assert outputs.requires_grad and inputs.requires_grad
        (grads,) = autograd.grad(
            outputs=(outputs.sum(),),
            inputs=(inputs,),
            create_graph=True,
            only_inputs=True,
        )
        penalty = grads.square().flatten(start_dim=-3).sum(dim=-1)
        loss = 0.5 * penalty.mean()
        return loss
