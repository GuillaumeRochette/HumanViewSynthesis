from math import sqrt, pi

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from nn.exponential_moving_average import ExponentialMovingAverage


def t_2d(x: Tensor, y: Tensor) -> Tensor:
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape

    a = torch.ones_like(x)
    b = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([a, b, x], dim=-1),
            torch.stack([b, a, y], dim=-1),
            torch.stack([b, b, a], dim=-1),
        ],
        dim=-2,
    )


def t_3d(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    assert x.ndim == y.ndim == z.ndim == 1
    assert x.shape == y.shape == z.shape

    a = torch.ones_like(x)
    b = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([a, b, b, x], dim=-1),
            torch.stack([b, a, b, y], dim=-1),
            torch.stack([b, b, a, z], dim=-1),
            torch.stack([b, b, b, a], dim=-1),
        ],
        dim=-2,
    )


def s_2d(x: Tensor, y: Tensor) -> Tensor:
    assert x.ndim == y.ndim == 1
    assert x.shape == y.shape

    a = torch.ones_like(x)
    b = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([x, b, b], dim=-1),
            torch.stack([b, y, b], dim=-1),
            torch.stack([b, b, a], dim=-1),
        ],
        dim=-2,
    )


def s_3d(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    assert x.ndim == y.ndim == z.ndim == 1
    assert x.shape == y.shape == z.shape

    a = torch.ones_like(x)
    b = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([x, b, b, b], dim=-1),
            torch.stack([b, y, b, b], dim=-1),
            torch.stack([b, b, z, b], dim=-1),
            torch.stack([b, b, b, a], dim=-1),
        ],
        dim=-2,
    )


def r_2d(t: Tensor) -> Tensor:
    assert t.ndim == 1

    a = torch.ones_like(t)
    b = torch.zeros_like(t)
    c = t.cos()
    s = t.sin()
    return torch.stack(
        [
            torch.stack([c, -s, b], dim=-1),
            torch.stack([s, c, b], dim=-1),
            torch.stack([b, b, a], dim=-1),
        ],
        dim=-2,
    )


def r_3d(t: Tensor, v: Tensor):
    assert t.ndim == v.ndim == 1
    x, y, z, _ = v.split(1, dim=-1)
    xx = x ** 2
    yy = y ** 2
    zz = z ** 2
    xy = x * y
    xz = x * z
    yz = y * z
    a = torch.ones_like(t)
    b = torch.zeros_like(t)
    c = t.cos()
    d = 1.0 - c
    s = t.sin()
    return torch.stack(
        [
            torch.stack([c + xx * d, xy * d - z * s, xz * d - y * s, b], dim=-1),
            torch.stack([xy * d + z * s, c + yy * d, yz * d - x * s, b], dim=-1),
            torch.stack([xz * d - y * s, yz * d + x * s, c + zz * d, b], dim=-1),
            torch.stack([b, b, b, a], dim=-1),
        ],
        dim=-2,
    )


class Augmentations(Module):
    def __init__(
        self,
        p: float,
        image_flip: bool = True,
        rotate: bool = True,
        translate: bool = True,
        brightness: bool = True,
        contrast: bool = True,
        luma_flip: bool = True,
        hue: bool = True,
        saturation: bool = True,
    ):
        super(Augmentations, self).__init__()
        self.register_buffer("p", torch.tensor(p))

        self.image_flip = image_flip
        self.rotate = rotate
        self.translate = translate
        self.brightness = brightness
        self.contrast = contrast
        self.luma_flip = luma_flip
        self.hue = hue
        self.saturation = saturation

        self.register_buffer("z", torch.zeros(1))
        self.register_buffer("id3", torch.eye(3)[None, ...])
        self.register_buffer("id4", torch.eye(4)[None, ...])
        v = torch.tensor([1.0, 1.0, 1.0, 0.0]) / sqrt(3)
        self.register_buffer("v", v)
        self.register_buffer("vv", v.outer(v))

        self.b_std = 0.2
        self.c_std = 0.5
        self.h_max = 1.0
        self.s_std = 1.0

    def sample_theta(self, input: Tensor) -> Tensor:
        n, c, h, w = input.shape
        z = self.z.expand(n)

        theta = self.id3

        if self.image_flip:
            p = torch.rand_like(z) < self.p
            x = 1.0 - 2.0 * torch.randint_like(z, low=0, high=2).where(p, z)
            y = torch.ones_like(x)
            theta = theta @ s_2d(x, y)

        if self.rotate:
            p = torch.rand_like(z) < self.p
            t = 0.5 * pi * torch.randint_like(z, low=0, high=4).where(p, z)
            theta = theta @ r_2d(t)

        if self.translate:
            p = torch.rand_like(z) < self.p
            x = (torch.rand_like(z) * 2.0 - 1.0).where(p, z)
            y = (torch.rand_like(z) * 2.0 - 1.0).where(p, z)
            theta = theta @ t_2d(x, y)

        return theta

    def sample_phi(self, input: Tensor) -> Tensor:
        n, c, h, w = input.shape
        z = self.z.expand(n)

        phi = self.id4

        if self.brightness:
            p = torch.rand_like(z) < self.p
            b = (torch.randn_like(z) * self.b_std).where(p, z)
            phi = phi @ t_3d(x=b, y=b, z=b)

        if self.contrast:
            p = torch.rand_like(z) < self.p
            q = (torch.randn_like(z) * self.c_std).exp2().where(p, 1.0 - z)
            phi = phi @ s_3d(q, q, q)

        if self.luma_flip:
            p = torch.rand_like(z) < self.p
            i = torch.randint_like(z, low=0, high=2).where(p, z)
            phi = phi @ (self.id4 - 2.0 * self.vv * i[..., None, None])

        if self.hue:
            p = torch.rand_like(z) < self.p
            t = ((torch.rand_like(z) * 2.0 - 1.0) * pi * self.h_max).where(p, z)
            phi = phi @ r_3d(t, self.v)

        if self.saturation:
            p = torch.rand_like(z) < self.p
            s = (torch.randn_like(z) * self.s_std).exp2().where(p, 1.0 - z)
            phi = phi @ (self.vv + (self.id4 - self.vv) * s[..., None, None])

        return phi

    def apply_theta(self, input: Tensor, theta: Tensor) -> Tensor:
        theta, _ = theta.split([2, 1], dim=-2)

        grid = F.affine_grid(
            theta=theta,
            size=input.shape,
            align_corners=True,
        )
        input = F.grid_sample(
            input=input,
            grid=((grid + 1.0) % 2.0) - 1.0,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )

        return input

    def apply_phi(self, input: Tensor, phi: Tensor) -> Tensor:
        n, c, h, w = input.shape
        phi, _ = phi.split([3, 1], dim=-2)
        A, b = phi.split([3, 1], dim=-1)

        input = input * 2.0 - 1.0
        input = input.reshape(n, c, h * w)
        input = A @ input + b
        input = input.reshape(n, c, h, w)
        input = (input + 1.0) * 0.5

        return input

    def forward(
        self,
        input: Tensor,
        theta: Tensor = None,
        phi: Tensor = None,
    ) -> Tensor:
        if theta is None:
            theta = self.sample_theta(input)

        if phi is None:
            phi = self.sample_phi(input)

        if (theta != self.id3).any():
            input = self.apply_theta(input, theta)

        if (phi != self.id4).any():
            input = self.apply_phi(input, phi)

        return input


class AdaptiveAugmentations(Augmentations):
    def __init__(
        self,
        p: float,
        r_target: float = 0.5,
        momentum: float = 0.8,
        alpha: float = 2.5e-3,
        image_flip: bool = True,
        rotate: bool = True,
        translate: bool = True,
        brightness: bool = True,
        contrast: bool = True,
        luma_flip: bool = True,
        hue: bool = True,
        saturation: bool = True,
    ):
        super(AdaptiveAugmentations, self).__init__(
            p=p,
            image_flip=image_flip,
            rotate=rotate,
            translate=translate,
            brightness=brightness,
            contrast=contrast,
            luma_flip=luma_flip,
            hue=hue,
            saturation=saturation,
        )
        r_target = torch.tensor(r_target)
        self.register_buffer("r_target", r_target)
        self.ema = ExponentialMovingAverage(r_target.shape, momentum=momentum)
        self.alpha = alpha

    @torch.no_grad()
    def update(self, r_current: Tensor):
        r_current = r_current.to(dtype=self.ema.average.dtype)
        a = self.alpha * (self.ema(r_current) - self.r_target).sign()
        self.p.copy_((self.p + a).clamp(min=0.0, max=1.0))
