import torch
from operations.losses.image import LPIPS
from torch import Tensor
from torchmetrics import Metric
from torch.nn import functional as F


def psnr(input: Tensor, target: Tensor, maximum: Tensor) -> Tensor:
    """
    Computes the PSNR between two images.

    :param input: [*, C, H, W]
    :param target: [*, C, H, W]
    :param maximum: []
    :return: [*]
    """
    return (
        20.0 * maximum.log10()
        - 10.0 * (input - target).square().mean(dim=[-3, -2, -1]).log10()
    )


class PSNRMetric(Metric):
    def __init__(self, maximum: float = 1.0, dist_sync_on_step=False):
        super(PSNRMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.register_buffer(
            "maximum",
            torch.tensor(maximum),
            persistent=False,
        )
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
        error = psnr(input=input, target=target, maximum=self.maximum)

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den


def gaussian_kernel(kernel_size: int, sigma: float):
    a = torch.arange(kernel_size, dtype=torch.float)
    a -= kernel_size // 2

    b = (-(a ** 2) / (2 * sigma ** 2)).exp()
    b /= b.sum()

    c = b[..., None] @ b[None, ...]
    return c


def ssim(input: Tensor, target: Tensor, kernel: Tensor, c1: float, c2: float) -> Tensor:
    if input.ndim == 3:
        input = input[None, ...]

    if target.ndim == 3:
        target = target[None, ...]

    x, y = input, target

    c = x.shape[-3]
    kernel = kernel.expand((c, 1) + kernel.shape)

    mu_x = F.conv2d(x, kernel, groups=c)
    mu_y = F.conv2d(y, kernel, groups=c)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x * x, kernel, groups=c) - mu_xx
    sigma_yy = F.conv2d(y * y, kernel, groups=c) - mu_yy
    sigma_xy = F.conv2d(x * y, kernel, groups=c) - mu_xy

    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    )
    ssim = ssim.mean(dim=[-3, -2, -1])
    return ssim


class SSIMMetric(Metric):
    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
        maximum_value: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        dist_sync_on_step=False,
    ):
        super(SSIMMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.register_buffer(
            "kernel",
            gaussian_kernel(kernel_size=kernel_size, sigma=sigma),
            persistent=False,
        )
        self.c1 = (k1 * maximum_value) ** 2
        self.c2 = (k2 * maximum_value) ** 2

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
        error = ssim(
            input=input,
            target=target,
            kernel=self.kernel,
            c1=self.c1,
            c2=self.c2,
        )

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den


class LPIPSMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super(LPIPSMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)

        self.extractor = LPIPS(net="alex")

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
        error = self.extractor(input, target)

        self.num += torch.sum(error)
        self.den += error.numel()

    def compute(self) -> Tensor:
        return self.num / self.den
