from typing import Tuple

from torch import Tensor

from data.Panoptic.skeleton import BODY_135_TO_BODY_117


def BODY_135_to_BODY_117_2d(p: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
    p = p[..., BODY_135_TO_BODY_117, :, :]
    c = c[..., BODY_135_TO_BODY_117, :, :]

    return p, c


def BODY_135_to_BODY_117_3d(p: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
    # Sternum replaces UpperNeck.
    p[..., 17, :, :] = p[..., [5, 6], :, :].mean(dim=-3)
    c[..., 17, :, :] = c[..., [5, 6], :, :].min(dim=-3)[0]

    # Head replaces HeadTop.
    p[..., 18, :, :] = p[..., [1, 2, 3, 4], :, :].mean(dim=-3)
    c[..., 18, :, :] = c[..., [1, 2, 3, 4], :, :].min(dim=-3)[0]

    p = p[..., BODY_135_TO_BODY_117, :, :]
    c = c[..., BODY_135_TO_BODY_117, :, :]

    return p, c
