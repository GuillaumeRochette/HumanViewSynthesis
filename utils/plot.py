from typing import Tuple, Union
from pathlib import Path

from colorsys import hls_to_rgb

import numpy as np
from skimage import draw

import torch
from torch import Tensor

from torchvision.transforms import functional as F


def get_palette(
    n: int,
    hue: float = 0.01,
    luminance: float = 0.6,
    saturation: float = 0.65,
) -> Tensor:
    hues = np.linspace(0, 1, n + 1)[:-1]
    hues += hue
    hues %= 1
    hues -= hues.astype(int)
    palette = [hls_to_rgb(float(hue), luminance, saturation) for hue in hues]
    palette = torch.tensor(palette)
    return palette


def _tensor_to_ndarray(x: Tensor) -> np.ndarray:
    x = x.cpu().numpy()
    return x


def _ndarray_to_tensor(x: np.ndarray) -> Tensor:
    x = torch.as_tensor(x)
    return x


def _draw_circles(
    image: Tensor,
    points: Tensor,
    colors: Tensor,
    radius: int = 1,
) -> Tensor:
    """

    :param image: [C, H, W]
    :param points: [N, 2, 1]
    :param colors: [N, C]
    :param radius:
    :return: [C, H, W]
    """
    assert points.shape[-2] == 2
    assert points.shape[-1] == 1

    _, h, w = image.shape
    n, _, _ = points.shape

    image = image.permute(1, 2, 0)

    cols, rows = points[:, 0, 0], points[:, 1, 0]

    cols = cols.round().long().clamp(min=0 - 1, max=w + 1)
    rows = rows.round().long().clamp(min=0 - 1, max=h + 1)

    image = _tensor_to_ndarray(image)
    cols = _tensor_to_ndarray(cols)
    rows = _tensor_to_ndarray(rows)
    colors = _tensor_to_ndarray(colors)

    for i in range(n):
        rr, cc, val = draw.circle_perimeter_aa(
            r=rows[i],
            c=cols[i],
            radius=radius,
        )
        draw.set_color(
            image=image,
            coords=(rr, cc),
            color=colors[i],
            alpha=val,
        )

    image = _ndarray_to_tensor(image)
    image = image.permute(2, 0, 1)

    return image


def _draw_lines(
    image: Tensor,
    start_points: Tensor,
    end_points: Tensor,
    colors: Tensor,
) -> Tensor:
    """

    :param image: [C, H, W]
    :param start_points: [N, 2, 1]
    :param end_points: [N, 2, 1]
    :param colors: [N, 3]
    :return: [C, H, W]
    """
    assert start_points.shape[-3] == end_points.shape[-3]
    assert start_points.shape[-2] == end_points.shape[-2] == 2
    assert start_points.shape[-1] == end_points.shape[-1] == 1

    _, h, w = image.shape
    n, _, _ = start_points.shape

    image = image.permute(1, 2, 0)

    start_cols, start_rows = start_points[:, 0, 0], start_points[:, 1, 0]
    end_cols, end_rows = end_points[:, 0, 0], end_points[:, 1, 0]

    start_cols = start_cols.round().long().clamp(min=0 - 1, max=w + 1)
    start_rows = start_rows.round().long().clamp(min=0 - 1, max=h + 1)
    end_cols = end_cols.round().long().clamp(min=0 - 1, max=w + 1)
    end_rows = end_rows.round().long().clamp(min=0 - 1, max=h + 1)

    image = _tensor_to_ndarray(image)
    start_cols = _tensor_to_ndarray(start_cols)
    start_rows = _tensor_to_ndarray(start_rows)
    end_cols = _tensor_to_ndarray(end_cols)
    end_rows = _tensor_to_ndarray(end_rows)
    colors = _tensor_to_ndarray(colors)

    for i in range(n):
        rr, cc, val = draw.line_aa(
            r0=start_rows[i],
            c0=start_cols[i],
            r1=end_rows[i],
            c1=end_cols[i],
        )
        draw.set_color(
            image=image,
            coords=(rr, cc),
            color=colors[i],
            alpha=val,
        )

    image = _ndarray_to_tensor(image)
    image = image.permute(2, 0, 1)

    return image


@torch.no_grad()
def draw_circles(
    image: Tensor,
    points: Tensor,
    colors: Tensor = None,
    radius: int = 1,
) -> Tensor:
    """

    :param image: [*, C, H, W]
    :param points: [*, N, 2, 1]
    :param colors: [*, N, C]
    :param radius:
    :return: [*, C, H, W]
    """

    assert image.shape[:-3] == points.shape[:-3]
    assert image.shape[-3] == 3
    assert points.shape[-2] == 2
    assert points.shape[-1] == 1

    shape = image.shape

    image = image.reshape((-1,) + image.shape[-3:])
    points = points.reshape((-1,) + points.shape[-3:])

    b, c, _, _ = image.shape
    _, n, _, _ = points.shape

    if colors is None:
        colors = get_palette(n)

    colors = colors.expand(b, n, c)

    for i in range(b):
        image[i] = _draw_circles(
            image=image[i],
            points=points[i],
            colors=colors[i],
            radius=radius,
        )

    image = image.reshape(shape)

    return image


@torch.no_grad()
def draw_lines(
    image: Tensor,
    start_points: Tensor,
    end_points: Tensor,
    colors: Tensor = None,
) -> Tensor:
    """

    :param image: [*, C, H, W]
    :param start_points: [*, N, 2, 1]
    :param end_points: [*, N, 2, 1]
    :param colors: [*, N, C]
    :return: [*, C, H, W]
    """
    assert image.shape[:-3] == start_points.shape[:-3] == end_points.shape[:-3]
    assert image.shape[-3] == 3
    assert start_points.shape[-3] == end_points.shape[-3]
    assert start_points.shape[-2] == end_points.shape[-2] == 2
    assert start_points.shape[-1] == end_points.shape[-1] == 1

    shape = image.shape

    image = image.reshape((-1,) + image.shape[-3:])
    start_points = start_points.reshape((-1,) + start_points.shape[-3:])
    end_points = end_points.reshape((-1,) + end_points.shape[-3:])

    b, c, _, _ = image.shape
    _, n, _, _ = start_points.shape

    if colors is None:
        colors = get_palette(n)

    colors = colors.expand(b, n, c)

    for i in range(b):
        image[i] = _draw_lines(
            image=image[i],
            start_points=start_points[i],
            end_points=end_points[i],
            colors=colors[i],
        )

    image = image.reshape(shape)

    return image


@torch.no_grad()
def draw_limbs(
    image: Tensor,
    points: Tensor,
    confidences: Tensor,
    edges: Tuple[Tuple[int, int]],
    colors: Tensor = None,
) -> Tensor:
    """

    :param image: [*, C, H, W]
    :param points: [*, N, 2, 1]
    :param confidences: [*, N, 1, 1]
    :param edges: [M, 2]
    :param colors: [*, M, C]
    :return: [*, C, H, W]
    """
    assert image.shape[:-3] == points.shape[:-3] == confidences.shape[:-3]
    assert image.shape[-3] == 3
    assert points.shape[-3] == confidences.shape[-3]
    assert points.shape[-2] == 2
    assert confidences.shape[-2] == 1
    assert points.shape[-1] == confidences.shape[-1] == 1

    start, end = list(zip(*edges))

    start_points = points[..., start, :, :]
    end_points = points[..., end, :, :]

    confidences = confidences[..., start, :, :] & confidences[..., end, :, :]
    start_points[~confidences.expand_as(start_points)] = -1.0
    end_points[~confidences.expand_as(end_points)] = -1.0

    return draw_lines(
        image=image,
        start_points=start_points,
        end_points=end_points,
        colors=colors,
    )


@torch.no_grad()
def show(image: Tensor):
    image = F.to_pil_image(image.cpu())
    image.show()


@torch.no_grad()
def save(image: Tensor, path: Union[str, Path]):
    if not isinstance(path, Path):
        path = Path(path)

    image = F.to_pil_image(image.cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
