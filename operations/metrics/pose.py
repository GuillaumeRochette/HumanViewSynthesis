import torch
from torch import Tensor
from torchmetrics import Metric

from operations.reduce import weighted_mean


def mpjpe(input: Tensor, target: Tensor) -> Tensor:
    """
    Computes the Mean Per Joint Position Error between a prediction and a target.

    :param input: [*, J, 3, 1].
    :param target: [*, J, 3, 1].
    :return: [*].
    """
    assert input.shape[:-2] == target.shape[:-2]
    assert input.shape[-1] == target.shape[-1] == 1

    error = (input - target).norm(p=2, dim=[-2, -1]).mean(dim=-1)
    return error


class MPJPEMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(MPJPEMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "num",
            default=torch.tensor(0, dtype=torch.double),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "den",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, input: Tensor, target: Tensor):
        error = mpjpe(input=input, target=target)

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den


def nmpjpe(input: Tensor, target: Tensor) -> Tensor:
    """
    Computes the Normalised Mean Per Joint Position Error between a prediction and a target.

    :param input: [*, J, 3, 1].
    :param target: [*, J, 3, 1].
    :return: [*].
    """
    assert input.shape[:-2] == target.shape[:-2]
    assert input.shape[-1] == target.shape[-1] == 1

    num = (input * target).sum(dim=[-3, -2, -1], keepdim=True)
    den = (input * input).sum(dim=[-3, -2, -1], keepdim=True)
    alpha = num / den
    input_ = alpha * input
    error = (input_ - target).norm(p=2, dim=[-2, -1]).mean(dim=-1)

    return error


class NMPJPEMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(NMPJPEMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "num",
            default=torch.tensor(0, dtype=torch.double),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "den",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, input: Tensor, target: Tensor):
        error = nmpjpe(input=input, target=target)

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den


def procrustes(input: Tensor, target: Tensor):
    """
    Computes the Procrustes transformation between a prediction and a target.

    :param input: Tensor of shape [*, J, 3, 1].
    :param target: Tensor of shape [*, J, 3, 1].
    :return:
    """
    input_ = input.squeeze(dim=-1)
    target_ = target.squeeze(dim=-1)

    input_mean = input_.mean(dim=-2)
    target_mean = target_.mean(dim=-2)

    input_ = input_ - input_mean[..., None, :]
    target_ = target_ - target_mean[..., None, :]

    input_norm = input_.norm(dim=[-2, -1])
    target_norm = target_.norm(dim=[-2, -1])

    input_ = input_ / input_norm[..., None, None]
    target_ = target_ / target_norm[..., None, None]

    A = target_.transpose(-1, -2) @ input_
    U, S, V = A.svd()

    tr = S.sum(dim=-1)
    a = tr * (target_norm / input_norm)

    R = U @ V.transpose(-1, -2)

    t = target_mean[..., None] - a[..., None, None] * (R @ input_mean[..., None])
    return a, R, t


def pmpjpe(input: Tensor, target: Tensor) -> Tensor:
    """
    Computes the Procrustes Mean Per Joint Position Error between a prediction and a target.

    :param input: [*, J, 3, 1].
    :param target: [*, J, 3, 1].
    :return:
    """
    assert input.shape[:-2] == target.shape[:-2]
    assert input.shape[-1] == target.shape[-1] == 1

    a, R, t = procrustes(input=input, target=target)
    a = a[..., None, None, None]
    R = R[..., None, :, :]
    t = t[..., None, :, :]
    input_ = a * (R @ input) + t
    error = (input_ - target).norm(p=2, dim=[-2, -1]).mean(dim=-1)

    return error


class PMPJPEMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(PMPJPEMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "num",
            default=torch.tensor(0, dtype=torch.double),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "den",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, input: Tensor, target: Tensor):
        error = pmpjpe(input=input, target=target)

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den


def mpvjpe(input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """
    Computes the Mean Per Valid Joint Position Error between a prediction and a target.

    :param input: [*, J, 3, 1].
    :param target: [*, J, 3, 1].
    :param weight: [*, J, 1, 1].
    :return:
    """
    assert input.shape[:-2] == target.shape[:-2] == weight.shape[:-2]
    assert input.shape[-1] == target.shape[-1] == 1
    assert weight.shape[-2] == weight.shape[-1] == 1

    error = weighted_mean(
        input=(input - target).norm(p=2, dim=[-2, -1], keepdim=True),
        weight=weight,
        dim=[-3, -2, -1],
    )

    return error


class MPVJPEMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(MPVJPEMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "num",
            default=torch.tensor(0, dtype=torch.double),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "den",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, input: Tensor, target: Tensor, weight: Tensor):
        error = mpvjpe(input=input, target=target, weight=weight)

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den
