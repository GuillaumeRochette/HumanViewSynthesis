from typing import Tuple, Union
from PIL import Image

import numpy as np

import torch
from torch import Tensor


class ArrayToTensor(object):
    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, x) -> Tensor:
        return torch.tensor(x, dtype=self.dtype)


class MaskToTensor(object):
    def __call__(self, x) -> Tensor:
        x = np.array(x, dtype=np.uint8)
        x = torch.as_tensor(x, dtype=torch.bool)
        x = x.unsqueeze(dim=-3)
        return x


class Threshold(object):
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x: Tensor) -> Tensor:
        return (x >= self.min_value) & (x <= self.max_value)


class StaticBoxCrop(object):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.height = self.width = size
        else:
            self.height, self.width = size

    def __call__(
        self,
        image: Image,
        bounds: Tuple[float, float, float, float],
    ):
        x_m, y_m, x_M, y_M = bounds
        assert x_m <= x_M and y_m <= y_M

        width, height = image.size

        assert self.width <= width and self.height <= height

        width, height = float(width), float(height)

        # Ensure the bounding box is within the image.
        x_m = max(x_m, 0.0)
        y_m = max(y_m, 0.0)
        x_M = min(x_M, width)
        y_M = min(y_M, height)

        # Compute the center of the unadjusted box.
        x_c = (x_m + x_M) / 2.0
        y_c = (y_m + y_M) / 2.0

        # Put the adjusted box around the center.
        x_m = x_c - self.width / 2.0
        y_m = y_c - self.height / 2.0
        x_M = x_c + self.width / 2.0
        y_M = y_c + self.height / 2.0

        # Offset the centered box if it goes outside the image.
        if x_m < 0.0:
            x_m, x_M = 0.0, x_M + (0.0 - x_m)
        elif x_M > width:
            x_m, x_M = x_m - (x_M - width), width

        if y_m < 0.0:
            y_m, y_M = 0.0, y_M + (0.0 - y_m)
        elif y_M > height:
            y_m, y_M = y_m - (y_M - height), height

        image = image.crop((x_m, y_m, x_M, y_M))
        return image, (x_m, y_m)


class DynamicSquareCrop(object):
    def __init__(self, margin: Union[float, Tuple[float, float]]):
        if isinstance(margin, float):
            self.margin_height = self.margin_width = margin
        else:
            self.margin_height, self.margin_width = margin

    def __call__(
        self,
        image: Image,
        bounds: Tuple[float, float, float, float],
    ):
        x_m, y_m, x_M, y_M = bounds
        assert x_m <= x_M and y_m <= y_M

        width, height = image.size
        width, height = float(width), float(height)

        # Ensure the bounding box is within the image.
        x_m = max(x_m, 0.0)
        y_m = max(y_m, 0.0)
        x_M = min(x_M, width)
        y_M = min(y_M, height)

        # Compute the center of the unadjusted box.
        x_c = (x_m + x_M) / 2.0
        y_c = (y_m + y_M) / 2.0

        d_x = (x_M - x_m) * (1.0 + self.margin_width)
        d_y = (y_M - y_m) * (1.0 + self.margin_height)

        d_x = min(d_x, width)
        d_y = min(d_y, height)

        d = max(d_x, d_y)

        # Put the adjusted box around the center.
        x_m = x_c - d / 2.0
        y_m = y_c - d / 2.0
        x_M = x_c + d / 2.0
        y_M = y_c + d / 2.0

        # Offset the centered box if it goes outside the image.
        if x_m < 0.0:
            x_m, x_M = 0.0, x_M + (0.0 - x_m)
        elif x_M > width:
            x_m, x_M = x_m - (x_M - width), width

        if y_m < 0.0:
            y_m, y_M = 0.0, y_M + (0.0 - y_m)
        elif y_M > height:
            y_m, y_M = y_m - (y_M - height), height

        image = image.crop((x_m, y_m, x_M, y_M))
        return image, (x_m, y_m)


class Resize(object):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.height = self.width = size
        else:
            self.height, self.width = size

    def __call__(self, image: Image):
        width, height = image.size
        width_ratio, height_ratio = self.width / width, self.height / height
        image = image.resize((self.width, self.height), Image.BILINEAR)
        return image, (width_ratio, height_ratio)


def stabilized_padding(
    crop_offset: Tensor,
    original_resolution: Tensor,
    cropped_resolution: Tensor,
    resized_resolution: Tensor,
) -> Tensor:
    x_off, y_off = crop_offset
    wo, ho = original_resolution
    wc, hc = cropped_resolution
    wr, hr = resized_resolution

    wp, hp = wo * wr // wc, ho * hr // hc
    if wp % 2 == 1:
        wp = wp + 1
    if hp % 2 == 1:
        hp = hp + 1

    left = x_off * wp / wo
    right = wp * (1 - x_off / wo) - wr
    upper = y_off * hp / ho
    lower = hp * (1 - y_off / ho) - hr

    padding = torch.stack([left, right, upper, lower]).round().long()
    return padding
