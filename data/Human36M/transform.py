from typing import Tuple, Union

import torch
from torchvision.transforms import ToTensor as ImageToTensor

from data.transforms import (
    MaskToTensor,
    DynamicSquareCrop,
    Resize,
    stabilized_padding,
)
from data.Human36M.skeleton import JOINTS
from data.Human36M.statistics import MEDIAN_PIXEL

from geometry.extrinsic import world_to_camera


class Human36MImageTransform(object):
    def __init__(
        self,
        cr_margin: float = None,
        re_size: Union[int, Tuple[int, int]] = None,
    ):
        self.i2t = ImageToTensor()
        self.m2t = MaskToTensor()

        self.crop = None
        if cr_margin is not None:
            self.crop = DynamicSquareCrop(margin=cr_margin)

        self.resize = None
        if re_size is not None:
            self.resize = Resize(size=re_size)

    def __call__(self, input: dict) -> dict:
        p = input["pose_2d"]
        c = torch.ones(len(JOINTS), 1, 1, dtype=torch.bool)
        input["pose_2d"] = {
            "p": p,
            "c": c,
        }

        p = input["pose_3d"]
        p = world_to_camera(
            xyz=p,
            R=input["R"][None, :, :],
            t=input["t"][None, :, :],
        )
        c = torch.ones(len(JOINTS), 1, 1, dtype=torch.bool)
        input["pose_3d"] = {
            "root": {
                "p": p[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                "c": c[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
            },
            "relative": {
                "p": p - p[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                "c": c & c[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
            },
        }

        x_off, y_off = 0.0, 0.0
        if self.crop:
            p, c = input["pose_2d"]["p"], input["pose_2d"]["c"]
            points = p[c.expand_as(p)].reshape(-1, 2, 1)
            points = points if len(points) > 0 else p
            (m, _), (M, _) = points.min(dim=-3), points.max(dim=-3)
            x_m, y_m = m.squeeze(dim=-1).tolist()
            x_M, y_M = M.squeeze(dim=-1).tolist()
            input["image"], (x_off, y_off) = self.crop(
                image=input["image"],
                bounds=(x_m, y_m, x_M, y_M),
            )
            input["mask"], (_, _) = self.crop(
                image=input["mask"],
                bounds=(x_m, y_m, x_M, y_M),
            )
            input["K"] = input["K"] - torch.tensor(
                [
                    [0.0, 0.0, x_off],
                    [0.0, 0.0, y_off],
                    [0.0, 0.0, 0.0],
                ]
            )
            # input["pose_2d"]["p"] = input["pose_2d"]["p"] - torch.tensor([[x_off], [y_off]])
        input["crop_offset"] = torch.tensor([x_off, y_off])
        input["cropped_resolution"] = torch.tensor(input["image"].size)

        if self.resize:
            input["image"], (w_r, h_r) = self.resize(image=input["image"])
            input["mask"], (_, _) = self.resize(image=input["mask"])
            input["K"] = input["K"] * torch.tensor(
                [
                    [w_r, 1.0, w_r],
                    [1.0, h_r, h_r],
                    [1.0, 1.0, 1.0],
                ]
            )
            # input["pose_2d"]["p"] = input["pose_2d"]["p"] * torch.tensor([[w_r], [h_r]])
        input["resized_resolution"] = torch.tensor(input["image"].size)

        input["stabilized_padding"] = stabilized_padding(
            crop_offset=input["crop_offset"],
            original_resolution=input["resolution"],
            cropped_resolution=input["cropped_resolution"],
            resized_resolution=input["resized_resolution"],
        )

        input["image"] = self.i2t(input["image"])
        input["mask"] = self.m2t(input["mask"])
        # input["masked_image"] = input["mask"] * input["image"]
        input["masked_image"] = (
            input["mask"] * input["image"]
            + ~input["mask"] * MEDIAN_PIXEL[..., None, None]
        )

        return input


class Human36MImagePairTransform(object):
    def __init__(
        self,
        cr_margin_A: Union[float, Tuple[float, float]] = None,
        cr_margin_B: Union[float, Tuple[float, float]] = None,
        re_size_A: Union[int, Tuple[int, int]] = None,
        re_size_B: Union[int, Tuple[int, int]] = None,
    ):
        self.i2t = ImageToTensor()
        self.m2t = MaskToTensor()

        self.views = ["A", "B"]
        self.crop = {
            "A": None,
            "B": None,
        }
        if cr_margin_A is not None:
            self.crop["A"] = DynamicSquareCrop(margin=cr_margin_A)
        if cr_margin_B is not None:
            self.crop["B"] = DynamicSquareCrop(margin=cr_margin_B)

        self.resize = {
            "A": None,
            "B": None,
        }
        if re_size_A is not None:
            self.resize["A"] = Resize(size=re_size_A)
        if re_size_B is not None:
            self.resize["B"] = Resize(size=re_size_B)

    def __call__(self, input: dict) -> dict:
        for v in self.views:
            p = input[v]["pose_2d"]
            c = torch.ones(len(JOINTS), 1, 1, dtype=torch.bool)
            input[v]["pose_2d"] = {
                "p": p,
                "c": c,
            }

            p = input["W"]["pose_3d"]
            p = world_to_camera(
                xyz=p,
                R=input[v]["R"][None, :, :],
                t=input[v]["t"][None, :, :],
            )
            c = torch.ones(len(JOINTS), 1, 1, dtype=torch.bool)
            input[v]["pose_3d"] = {
                "root": {
                    "p": p[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                    "c": c[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                },
                "relative": {
                    "p": p - p[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                    "c": c & c[JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                },
            }

            x_off, y_off = 0.0, 0.0
            if self.crop[v]:
                p, c = input[v]["pose_2d"]["p"], input[v]["pose_2d"]["c"]
                points = p[c.expand_as(p)].reshape(-1, 2, 1)
                points = points if len(points) > 0 else p
                (m, _), (M, _) = points.min(dim=-3), points.max(dim=-3)
                x_m, y_m = m.squeeze(dim=-1).tolist()
                x_M, y_M = M.squeeze(dim=-1).tolist()
                input[v]["image"], (x_off, y_off) = self.crop[v](
                    image=input[v]["image"],
                    bounds=(x_m, y_m, x_M, y_M),
                )
                input[v]["mask"], (_, _) = self.crop[v](
                    image=input[v]["mask"],
                    bounds=(x_m, y_m, x_M, y_M),
                )
                input[v]["K"] = input[v]["K"] - torch.tensor(
                    [
                        [0.0, 0.0, x_off],
                        [0.0, 0.0, y_off],
                        [0.0, 0.0, 0.0],
                    ]
                )
                # input[v]["pose_2d"]["p"] = input[v]["pose_2d"]["p"] - torch.tensor([[x_off], [y_off]])
            input[v]["crop_offset"] = torch.tensor([x_off, y_off])
            input[v]["cropped_resolution"] = torch.tensor(input[v]["image"].size)

            if self.resize[v]:
                input[v]["image"], (w_r, h_r) = self.resize[v](image=input[v]["image"])
                input[v]["mask"], (_, _) = self.resize[v](image=input[v]["mask"])
                input[v]["K"] = input[v]["K"] * torch.tensor(
                    [
                        [w_r, 1.0, w_r],
                        [1.0, h_r, h_r],
                        [1.0, 1.0, 1.0],
                    ]
                )
                # input[v]["pose_2d"]["p"] = input[v]["pose_2d"]["p"] * torch.tensor([[w_r], [h_r]])
            input[v]["resized_resolution"] = torch.tensor(input[v]["image"].size)

            input[v]["stabilized_padding"] = stabilized_padding(
                crop_offset=input[v]["crop_offset"],
                original_resolution=input[v]["resolution"],
                cropped_resolution=input[v]["cropped_resolution"],
                resized_resolution=input[v]["resized_resolution"],
            )

            input[v]["image"] = self.i2t(input[v]["image"])
            input[v]["mask"] = self.m2t(input[v]["mask"])
            # input[v]["masked_image"] = input[v]["mask"] * input[v]["image"]
            input[v]["masked_image"] = (
                input[v]["mask"] * input[v]["image"]
                + ~input[v]["mask"] * MEDIAN_PIXEL[..., None, None]
            )

        return input
