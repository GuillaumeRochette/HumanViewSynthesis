from typing import Union, Tuple

from math import pi, log10

import torch
from torch import Tensor
from torch.nn import Module

from geometry import intrinsic, vector


def get_pixel_grid(h: int, w: int) -> Tensor:
    """
    Creates a pixel grid containing pixels, e.g. pixel_grid[i, j] = [j, i, 1.0].

    :param h:
    :param w:
    :return: [H, W, 3, 1]
    """
    cols, rows = torch.arange(w, dtype=torch.float), torch.arange(h, dtype=torch.float)
    x, y = torch.meshgrid([cols, rows], indexing="xy")
    z = torch.ones_like(x)
    xyz = torch.stack([x, y, z], dim=-1).unsqueeze(dim=-1)

    return xyz


def pixel_grid_to_ray_grid(xyz: Tensor, K: Tensor, dist_coef: Tensor = None) -> Tensor:
    """
    Converts a pixel grid to a ray grid given camera intrinsic parameters and distortion coefficients.

    :param xyz: [H, W, 3, 1]
    :param K: [*, 3, 3]
    :param dist_coef: [*, D]
    :return: [*, H, W, 3, 1]
    """
    shape = K.shape[:-2]
    xyz = xyz.expand(shape + xyz.shape)

    xy, z = xyz.split([2, 1], dim=-2)

    xy = intrinsic.pixels_to_rays(xy=xy, K=K[..., None, None, :, :])
    if dist_coef is not None:
        xy = intrinsic.undistort(xy=xy, dist_coef=dist_coef[..., None, None, :])

    xyz = torch.cat([xy, z], dim=-2)

    xyz = vector.normalize(xyz)

    return xyz


def invert_lambd_naive(lambd: Tensor, alpha: float, eps: float) -> Tuple[Tensor, ...]:
    """
    Invert lambd in a naive way, which is less stable numerically when computing the integrals.

    :param lambd: [*, N, 3, 1]
    :param alpha:
    :param eps:
    :return: [*, N, 3, 1], []
    """
    lambd = 1.0 / lambd.clamp(min=eps)
    alpha = torch.tensor(alpha, device=lambd.device)

    return lambd, alpha


def symmetric_bilinear_form(x: Tensor, A: Tensor, y: Tensor) -> Tensor:
    """
    Computes a symmetric bilinear form.

    :param x: [*, N, 1]
    :param A: [*, N, N].
    :param y: [*, N, 1]
    :return: [*]
    """
    return (x.transpose(-1, -2) @ A @ y).flatten(start_dim=-3)


def compute_quantities_naive(
    rays: Tensor, mu: Tensor, rho: Tensor, lambd: Tensor
) -> Tuple[Tensor, ...]:
    """
    Computes the "big" symmetric bilinear forms, in a naive way. This is slow but mathematically simpler.

    :param pixels: [H, W, 3, 1]
    :param mu: [*, N, 3, 1]
    :param rho: [*, N, 3, 3]
    :param lambd: [*, N, 3, 1]
    :return:
    """

    rays = rays[..., None, :, :]
    mu = mu[..., None, None, :, :, :]
    sigma = rho @ (torch.eye(3) * lambd) @ rho.transpose(-1, -2)
    sigma = sigma[..., None, None, :, :, :]

    rays_sigma_rays = symmetric_bilinear_form(x=rays, A=sigma, y=rays)
    mu_sigma_mu = symmetric_bilinear_form(x=mu, A=sigma, y=mu)
    rays_sigma_mu = symmetric_bilinear_form(x=rays, A=sigma, y=mu)

    return rays_sigma_rays, mu_sigma_mu, rays_sigma_mu


def invert_lambd(lambd: Tensor, alpha: float, eps: float) -> Tuple[Tensor, ...]:
    """
    Invert lambd, while transferring its magnitude into the alpha coefficient, in order to improve numerical stability.
    Please refer to the equation (69) in the appendix, where we embed the smallest value of lambd into alpha.

    :param lambd: [*, N, 3, 1]
    :param alpha:
    :param eps:
    :return: [*, N, 3, 1], [*, 1, 1, 1]
    """
    logscale = lambd.flatten(start_dim=-3).min(dim=-1)[0].clamp(min=1e-6).log10().ceil()
    scale = (10 ** logscale)[..., None, None, None]
    lambd = scale / lambd.clamp(min=eps)
    alpha = alpha * scale

    return lambd, alpha


def compute_quantities(
    rays: Tensor, mu: Tensor, rho: Tensor, lambd: Tensor
) -> Tuple[Tensor, ...]:
    """
    Computes the "big" symmetric bilinear forms, in an optimized way, using the fact that lambd is a diagonal matrix.
    Therefore we can compute only two "big" matrix multiplications, and then simply using scalar multiplications and sums.
    This is about much faster, especially when the images grow bigger.

    :param pixels: [H, W, 3, 1]
    :param mu: [*, N, 3, 1]
    :param rho: [*, N, 3, 3]
    :param lambd: [*, N, 3, 1]
    :return:
    """
    rays = rays[..., None, :, :]
    mu = mu[..., None, None, :, :, :]
    rho = rho[..., None, None, :, :, :]
    lambd = lambd[..., None, None, :, :, :]

    rho_rays = rho.transpose(-1, -2) @ rays
    rho_mu = rho.transpose(-1, -2) @ mu

    rays_sigma_rays = (lambd * rho_rays.square()).sum(dim=[-2, -1])
    mu_sigma_mu = (lambd * rho_mu.square()).sum(dim=[-2, -1])
    rays_sigma_mu = (lambd * rho_rays * rho_mu).sum(dim=[-2, -1])

    return rays_sigma_rays, mu_sigma_mu, rays_sigma_mu


def optimal_z(rays_sigma_mu: Tensor, rays_sigma_rays: Tensor, eps: float) -> Tensor:
    """
    Refer to equation (82) in the appendix.

    :param rays_sigma_mu:
    :param rays_sigma_rays:
    :param eps:
    :return:
    """
    return rays_sigma_mu / rays_sigma_rays.clamp(min=eps)


def max_z(z: Tensor) -> Tensor:
    """
    Refer to equation (85) in the appendix.

    :param z:
    :return:
    """
    return z.flatten(start_dim=-3).max(dim=-1)[0][..., None, None, None]


def normalize_weights(weights: Tensor, eps: float) -> Tensor:
    """
    Refer to equation (43) in the appendix.

    :param weights:
    :param eps:
    :return:
    """
    return weights / weights.sum(dim=-1, keepdim=True).clamp(min=eps)


def splat_image(weights: Tensor, appearance: Tensor) -> Tensor:
    """
    Refer to equation (44) in the appendix.

    :param weights:
    :param appearance:
    :return:
    """
    image = weights @ appearance[..., None, :, :]
    dims = [i for i in range(image.ndim)]
    image = image.permute(dims=dims[:-3] + dims[-1:] + dims[-3:-1]).contiguous()

    return image


def density(x: Tensor) -> Tensor:
    # density = (-x.square()).exp()
    density = 1.0 / (1.0 + x.pow(4))

    return density


def integral(
    rays_sigma_rays: Tensor,
    mu_sigma_mu: Tensor,
    rays_sigma_mu: Tensor,
    alpha: Tensor,
    eps: float,
) -> Tensor:
    a = 0.5 * (pi * alpha).sqrt()
    b = rays_sigma_rays.clamp(min=0.0).sqrt().clamp(min=eps)
    c = (rays_sigma_mu / (alpha.sqrt() * b)).erfc().clamp(min=1.0, max=1.0)
    d = mu_sigma_mu - (rays_sigma_mu.square() / rays_sigma_rays.clamp(min=eps))
    e = (-d / alpha).clamp(max=log10(1.0 / eps)).exp().clamp(max=1.0 / eps)
    integral = a / b * c * e

    return integral.clamp(min=0.0)


def background_integral(z: Tensor, alpha: Tensor) -> Tensor:
    a = 0.5 * (pi * alpha).sqrt()
    b = (z / alpha.sqrt()).erfc().clamp(min=1.0, max=1.0)
    background_integral = a * b

    return background_integral.clamp(min=0.0)


def integral_raycasting(
    pixels: Tensor,
    mu: Tensor,
    rho: Tensor,
    lambd: Tensor,
    appearance: Tensor,
    background_appearance: Tensor,
    K: Tensor,
    dist_coef: Tensor = None,
    alpha: float = 2.5e-2,
    beta: float = 2e0,
    eps: float = 1e-8,
) -> Tensor:
    """

    :param pixels: [H, W, 3, 1]
    :param mu: [*, N, 3, 1]
    :param rho: [*, N, 3, 3]
    :param lambd: [*, N, 3, 1]
    :param appearance: [*, N, 3]
    :param background_appearance: [*, 1, 3]
    :param K: [*, 3, 3]
    :param dist_coef: [*, D]
    :param alpha:
    :param beta:
    :param function:
    :param eps:
    :return:
    """

    rays = pixel_grid_to_ray_grid(
        xyz=pixels,
        K=K,
        dist_coef=dist_coef,
    )

    lambd, alpha = invert_lambd(
        lambd=lambd,
        alpha=alpha,
        eps=eps,
    )

    rays_sigma_rays, mu_sigma_mu, rays_sigma_mu = compute_quantities(
        rays=rays,
        mu=mu,
        rho=rho,
        lambd=lambd,
    )

    z = optimal_z(rays_sigma_mu=rays_sigma_mu, rays_sigma_rays=rays_sigma_rays, eps=eps)

    z_background = beta * max_z(z=z)

    weights = density(x=z) * integral(
        rays_sigma_rays=rays_sigma_rays,
        mu_sigma_mu=mu_sigma_mu,
        rays_sigma_mu=rays_sigma_mu,
        alpha=alpha,
        eps=eps,
    )

    weight_background = density(x=z_background) * background_integral(
        z=z_background,
        alpha=alpha,
    )
    shape = weights.shape[:-1] + weight_background.shape[-1:]
    weight_background = weight_background.expand(shape)

    weights = torch.cat([weights, weight_background], dim=-1)
    weights = normalize_weights(weights=weights, eps=eps)

    appearance = torch.cat([appearance, background_appearance], dim=-2)

    image = splat_image(weights=weights, appearance=appearance)

    return image


class IntegralRayCasting(Module):
    def __init__(
        self,
        resolution: Union[int, Tuple[int, int]],
        alpha: float = 2.5e-2,
        beta: float = 2e0,
    ):
        super(IntegralRayCasting, self).__init__()
        assert alpha > 0.0
        assert beta >= 1.0

        if isinstance(resolution, int):
            h = w = resolution
        else:
            h, w = resolution

        self.register_buffer("pixels", get_pixel_grid(h=h, w=w), persistent=False)

        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        mu: Tensor,
        rho: Tensor,
        lambd: Tensor,
        appearance: Tensor,
        background_appearance: Tensor,
        K: Tensor,
        dist_coef: Tensor = None,
    ) -> Tensor:
        """
        Forward pass.

        :param mu: [*, N, 3, 1]
        :param rho: [*, N, 3, 3]
        :param lambd: [*, N, 3, 3]
        :param a: [*, N, A]
        :param b: [*, A]
        :param K: [*, 3, 3]
        :param dist_coef: [*, D]
        :return: [*, A, H, W]
        """
        return integral_raycasting(
            pixels=self.pixels,
            mu=mu,
            rho=rho,
            lambd=lambd,
            appearance=appearance,
            background_appearance=background_appearance,
            K=K,
            dist_coef=dist_coef,
            alpha=self.alpha,
            beta=self.beta,
        )
