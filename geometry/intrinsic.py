from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def projection_to_rays(xyz: Tensor, min_z: float = 1e-4):
    """
    Projects 3D coordinates on rays.

    :param xyz: 3D coordinates in of shape [*, 3, 1].
    :param min_z: Minimal depth for the 3D coordinates.
    :return: 2D coordinates in the ray space of shape [*, 2, 1].
    """
    assert xyz.shape[-2:] == (3, 1)

    xy, z = xyz.split([2, 1], dim=-2)
    xy = xy / z.clamp(min=min_z)

    return xy


def _unpack_dist_coef(dist_coef: Tensor) -> Tuple[Tensor, ...]:
    """
    Unpacks distortion coefficients.

    :param dist_coef: Distortion coefficients of shape [*, D].
    :return: D distortion coefficients of shape [*, 1, 1].
    """
    assert dist_coef.shape[-1] in [4, 5, 8, 12]

    dist_coef = F.pad(dist_coef, (0, 12 - dist_coef.shape[-1]))
    dist_coef = dist_coef.unsqueeze(dim=-1).split(1, dim=-2)

    return dist_coef


def _radial_distortion(
    r2: Tensor,
    k1: Tensor,
    k2: Tensor,
    k3: Tensor,
    k4: Tensor,
    k5: Tensor,
    k6: Tensor,
) -> Tensor:
    """
    Computes the radial distortion.

    :param r2: 2D norms of shape [*, 1, 1].
    :param k1, k2, k3, k4, k5, k6: Radial distortion coefficient of shape [*, 1, 1].
    :return: Radial distortion of shape [*, 1, 1].
    """
    num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    rad = num / den

    return rad


def _tangential_distortion(
    x: Tensor,
    y: Tensor,
    r2: Tensor,
    p1: Tensor,
    p2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the tangential distortion.

    :param x: X-axis coordinates of shape [*, 1, 1].
    :param y: Y-axis coordinates of shape [*, 1, 1].
    :param r2: 2D norms of shape [*, 1, 1].
    :param p1, p2: Tangential distortion coefficient of shape [*, 1, 1].
    :return: Tangential distortion of shape [*, 1, 1].
    """

    tan_x = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    tan_y = 2 * p2 * x * y + p1 * (r2 + 2 * y ** 2)

    return tan_x, tan_y


def _thin_prism_distortion(
    r2: Tensor,
    s1: Tensor,
    s2: Tensor,
    s3: Tensor,
    s4: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the thin prism distortion.

    :param r2: 2D norms of shape [*, 1, 1].
    :param s1, s2, s3, s4: Thin prism distortion coefficient of shape [*, 1, 1].
    :return: Thin prism distortion of shape [*, 1, 1].
    """

    thp_x = r2 * (s1 + r2 * s2)
    thp_y = r2 * (s3 + r2 * s4)

    return thp_x, thp_y


def distort(xy: Tensor, dist_coef: Tensor) -> Tensor:
    """
    Distorts 2D coordinates in the ray space ([-1, 1]x[-1, 1]) given distortion coefficients of the lens.

    :param xy: Undistorted 2D coordinates in the ray space of shape [*, 2, 1].
    :param dist_coef: Distortion coefficients of shape [*, D].
    :return: Distorted 2D coordinates in the ray space of shape [*, 2, 1].
    """
    assert xy.ndim - 1 == dist_coef.ndim
    assert xy.shape[-2:] == (2, 1)
    assert dist_coef.shape[-1] in [4, 5, 8, 12]

    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = _unpack_dist_coef(dist_coef)
    x, y = xy.split(1, dim=-2)

    r2 = x ** 2 + y ** 2
    rad = _radial_distortion(r2, k1, k2, k3, k4, k5, k6)
    tan_x, tan_y = _tangential_distortion(x, y, r2, p1, p2)
    thp_x, thp_y = _thin_prism_distortion(r2, s1, s2, s3, s4)

    x = x * rad + tan_x + thp_x
    y = y * rad + tan_y + thp_y

    xy = torch.cat([x, y], dim=-2)

    return xy


def undistort(xy: Tensor, dist_coef: Tensor, max_iterations: int = 5) -> Tensor:
    """
    Undistorts 2D coordinates in the ray space ([-1, 1]x[-1, 1]) given distortion coefficients of the lens.
    It is a batched and differentiable implementation of OpenCV's "undistort" iterative algorithm.
    See https://github.com/opencv/opencv/blob/7de627c504a3b8aa16cdc3de56efdc358df4b061/modules/calib3d/src/undistort.dispatch.cpp#L361

    :param xy: Distorted 2D coordinates in the ray space of shape [*, 2, 1].
    :param dist_coef: Distortion coefficients of shape [*, D].
    :param max_iterations: Maximum number of iterations.
    :return: Undistorted 2D coordinates in the ray space of shape [*, 2, 1].
    """
    assert xy.ndim - 1 == dist_coef.ndim
    assert xy.shape[-2:] == (2, 1)
    assert dist_coef.shape[-1] in [4, 5, 8, 12]

    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = _unpack_dist_coef(dist_coef)
    x, y = xy.split(1, dim=-2)
    x0, y0 = x.clone(), y.clone()

    for _ in range(max_iterations):
        r2 = x ** 2 + y ** 2
        rad = _radial_distortion(r2, k1, k2, k3, k4, k5, k6)
        tan_x, tan_y = _tangential_distortion(x, y, r2, p1, p2)
        thp_x, thp_y = _thin_prism_distortion(r2, s1, s2, s3, s4)

        m = rad > 0
        x = m * (x0 - tan_x - thp_x) / rad + ~m * x0
        y = m * (y0 - tan_y - thp_y) / rad + ~m * y0

    xy = torch.cat([x, y], dim=-2)

    return xy


def _unpack_intrinsic_parameters(K: Tensor) -> Tuple[Tensor, ...]:
    """
    Unpacks camera intrinsic parameters.

    :param K: Camera intrinsic parameters of shape [*, 3, 3].
    :return: 4 Camera intrinsic parameters of [*, 1, 1].
    """
    assert K.shape[-2:] == (3, 3)
    f_x, _, c_x, _, f_y, c_y, _, _, _ = (
        K.flatten(start_dim=-2).unsqueeze(dim=-2).split(1, dim=-1)
    )

    return f_x, f_y, c_x, c_y


def rays_to_pixels(xy: Tensor, K: Tensor) -> Tensor:
    """
    Converts 2D coordinates from the ray space ([-1, 1]x[-1, 1]) to the pixel space ([0..H]x[0..W]).

    :param xy: 2D coordinates in the ray space of shape [*, 2, 1].
    :param K: Camera intrinsic parameters of shape [*, 3, 3].
    :return: 2D coordinates in the pixel space of shape [*, 2, 1].
    """
    assert xy.ndim == K.ndim
    assert xy.shape[-2:] == (2, 1)
    assert K.shape[-2:] == (3, 3)

    x, y = xy.split(1, dim=-2)
    f_x, f_y, c_x, c_y = _unpack_intrinsic_parameters(K)
    x = f_x * x + c_x
    y = f_y * y + c_y
    xy = torch.cat([x, y], dim=-2)

    return xy


def pixels_to_rays(xy: Tensor, K: Tensor) -> Tensor:
    """
    Converts 2D coordinates from the pixel space ([0..H]x[0..W]) to the ray space ([-1, 1]x[-1, 1]).

    :param xy: 2D coordinates in the pixel space of shape [*, 2, 1].
    :param K: Camera intrinsic parameters of shape [*, 3, 3].
    :return: 2D coordinates in the ray space of shape [*, 2, 1].
    """
    assert xy.ndim == K.ndim
    assert xy.shape[-2:] == (2, 1)
    assert K.shape[-2:] == (3, 3)

    x, y = xy.split(1, dim=-2)
    f_x, f_y, c_x, c_y = _unpack_intrinsic_parameters(K)
    x = (x - c_x) / f_x
    y = (y - c_y) / f_y
    xy = torch.cat([x, y], dim=-2)

    return xy


def perspective_projection(
    xyz: Tensor,
    K: Tensor = None,
    dist_coef: Tensor = None,
) -> Tensor:
    """
    Projects 3D coordinates onto 2D, applying distortion and intrinsics if provided.

    :param xyz: 3D coordinates in of shape [*, 3, 1].
    :param K: Camera intrinsic parameters of shape [*, 3, 3].
    :param dist_coef: Distortion coefficients of shape [*, D].
    :return: 2D coordinates of shape [*, 2, 1].
    """
    assert xyz.shape[-2:] == (3, 1)

    xy = projection_to_rays(xyz=xyz)

    if dist_coef is not None:
        xy = distort(xy=xy, dist_coef=dist_coef)

    if K is not None:
        xy = rays_to_pixels(xy=xy, K=K)

    return xy


def rescale_pixels(
    xy: Tensor,
    old_resolution: Tensor,
    new_resolution: Tensor,
) -> Tensor:
    """
    Resize 2D pixel coordinates.

    :param xy: 2D coordinates in the pixel space of shape [*, 2, 1].
    :param old_resolution: Old resolution of shape [*, 2].
    :param new_resolution: New resolution of shape [*, 2].
    :return: Resized 2D coordinates in the pixel space of shape [*, 2, 1].
    """
    assert xy.ndim - 1 == old_resolution.ndim == new_resolution.ndim
    assert xy.shape[-2:] == (2, 1)
    assert old_resolution.shape[-1:] == new_resolution.shape[-1:] == (2,)

    ratio = new_resolution / old_resolution
    xy = xy * ratio.unsqueeze(dim=-1)

    return xy
