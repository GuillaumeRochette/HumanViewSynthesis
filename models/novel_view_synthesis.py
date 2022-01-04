from typing import Tuple
from munch import Munch, RecursiveMunch
from math import log2

import torch
from torch import Tensor
from torch.nn import Module

from geometry.extrinsic import world_to_camera, camera_to_camera
from geometry.primitives import Pose3DToLimbs
from rendering.integral_raycasting import IntegralRayCasting

from models.pose_2d_to_pose_3d import Pose2DToPose3D
from models.image_to_appearance import ImageToAppearance
from models.image_to_image import ImageToImage


class NovelViewSynthesis(Module):
    def __init__(
        self,
        n_joints: int,
        d_embedding: int,
        dropout: float,
        rendering_resolution: int,
        target_resolution: int,
        edges: Tuple[Tuple[int, int]],
        widths: Tensor,
        appearance_features: int,
        base_channels: int,
    ):
        super(NovelViewSynthesis, self).__init__()

        self.p2p = Pose2DToPose3D(
            n_joints=n_joints,
            d_embedding=d_embedding,
            dropout=dropout,
        )

        self.i2a = ImageToAppearance(out_dims=(len(edges), appearance_features))

        self.p2l = Pose3DToLimbs(edges=edges, widths=widths)

        self.renderer = IntegralRayCasting(resolution=rendering_resolution)
        self.register_buffer(
            "background",
            torch.zeros(1, appearance_features),
            persistent=False,
        )

        outer_blocks = int(log2(target_resolution / rendering_resolution))

        s_r = 2.0 ** -outer_blocks
        s_K = torch.tensor(
            [
                [s_r, 1.0, s_r],
                [1.0, s_r, s_r],
                [1.0, 1.0, 1.0],
            ]
        )
        self.register_buffer("s_K", s_K, persistent=False)

        self.r2i = ImageToImage(
            inner_blocks=4,
            mid_blocks=2,
            outer_blocks=outer_blocks,
            in_channels=appearance_features,
            out_channels=3,
            style_channels=len(edges) * appearance_features,
            base_channels=base_channels,
        )

    def forward(self, input: Munch) -> Munch:
        return self.infer_real(input=input)

    def infer_real(self, input: Munch) -> Munch:
        """
        From an input view, it infers a novel image tied to an existing viewpoint.

        :param input: Munch containing:
            - input.A.image
            - input.A.pose_2d.p
            - input.A.pose_2d.c
            - input.A.resolution
            - input.A.pose_3d.root.p
            - input.A.R, input.A.t
            - input.B.R, input.B.t
            - input.B.K, input.B.dist_coef
        :return: output: Munch containing:
            - output.A.appearance
            - output.A.pose_3d.relative.p
            - output.B.pose_3d.p
            - output.B.mu, output.B.rho, output.B.lambd
            - output.B.render
            - output.B.image
        """
        output = RecursiveMunch()

        output.A.pose_3d.relative.p = self.p2p(
            p=input.A.pose_2d.p,
            c=input.A.pose_2d.c,
            resolution=input.A.resolution,
        )

        output.A.pose_3d.p = input.A.pose_3d.root.p + output.A.pose_3d.relative.p

        output.A.appearance = self.i2a(input.A.image)

        output.B.pose_3d.p = camera_to_camera(
            xyz=output.A.pose_3d.p,
            R1=input.A.R[..., None, :, :],
            t1=input.A.t[..., None, :, :],
            R2=input.B.R[..., None, :, :],
            t2=input.B.t[..., None, :, :],
        )

        output.B.mu, output.B.rho, output.B.lambd = self.p2l(output.B.pose_3d.p)

        output.B.render = self.renderer(
            mu=output.B.mu,
            rho=output.B.rho,
            lambd=output.B.lambd,
            appearance=output.A.appearance,
            background_appearance=self.background.expand(
                output.A.appearance.shape[:-2] + self.background.shape
            ),
            K=self.s_K * input.B.K,
            dist_coef=input.B.dist_coef,
        )

        output.B.image = self.r2i(
            output.B.render,
            output.A.appearance.flatten(start_dim=-2),
        )

        return output

    def infer_virtual(self, input: Munch) -> Munch:
        """
        From an input view, it infers a novel image with respect to a virtual viewpoint.

        :param input: Munch containing:
            - input.image
            - input.pose_3d.p
            - input.R, input.t
            - input.K, input.dist_coef
        :return: output: Munch containing:
            - output.appearance
            - output.pose_3d.p
            - output.mu, output.rho, output.lambd
            - output.render
            - output.image
        """
        output = RecursiveMunch()

        output.pose_3d.relative.p = self.p2p(
            p=input.pose_2d.p,
            c=input.pose_2d.c,
            resolution=input.resolution,
        )

        output.pose_3d.p = input.pose_3d.root.p + output.pose_3d.relative.p

        output.appearance = self.i2a(input.image)

        output.pose_3d.p = world_to_camera(
            xyz=output.pose_3d.p,
            R=input.R[..., None, :, :],
            t=input.t[..., None, :, :],
        )

        output.mu, output.rho, output.lambd = self.p2l(output.pose_3d.p)

        output.render = self.renderer(
            mu=output.mu,
            rho=output.rho,
            lambd=output.lambd,
            appearance=output.appearance,
            background_appearance=self.background.expand(
                output.appearance.shape[:-2] + self.background.shape
            ),
            K=self.s_K * input.K,
            dist_coef=input.dist_coef,
        )

        output.image = self.r2i(
            output.render,
            output.appearance.flatten(start_dim=-2),
        )

        return output

    def motion_transfer(self, input: Munch) -> Munch:
        """
        From two input images, it extracts the pose and appearance respectively,
         then infers a novel image tied to an existing viewpoint.
        :param input: Munch containing:
            - input.C.image
            - input.A.pose_3d.p
            - input.A.R, input.A.t
            - input.B.R, input.B.t
            - input.B.K, input.B.dist_coef
        :return: output: Munch containing:
            - output.C.appearance
            - output.B.pose_3d.p
            - output.B.mu, output.B.rho, output.B.lambd
            - output.B.render
            - output.B.image
        """
        output = RecursiveMunch()

        output.A.pose_3d.relative.p = self.p2p(
            p=input.A.pose_2d.p,
            c=input.A.pose_2d.c,
            resolution=input.A.resolution,
        )

        output.A.pose_3d.p = input.A.pose_3d.root.p + output.A.pose_3d.relative.p

        output.C.appearance = self.i2a(input.C.image)

        output.B.pose_3d.p = camera_to_camera(
            xyz=output.A.pose_3d.p,
            R1=input.A.R[..., None, :, :],
            t1=input.A.t[..., None, :, :],
            R2=input.B.R[..., None, :, :],
            t2=input.B.t[..., None, :, :],
        )

        output.B.mu, output.B.rho, output.B.lambd = self.p2l(output.B.pose_3d.p)

        output.B.render = self.renderer(
            mu=output.B.mu,
            rho=output.B.rho,
            lambd=output.B.lambd,
            appearance=output.C.appearance,
            background_appearance=self.background.expand(
                output.C.appearance.shape[:-2] + self.background.shape
            ),
            K=self.s_K * input.B.K,
            dist_coef=input.B.dist_coef,
        )

        output.B.image = self.r2i(
            output.B.render,
            output.C.appearance.flatten(start_dim=-2),
        )

        return output


if __name__ == "__main__":
    from pprint import pprint

    from data.Panoptic.skeleton import JOINTS, EDGES
    from data.Panoptic.statistics import WIDTHS
    from utils.logging import summarize_dict

    m = NovelViewSynthesis(
        n_joints=len(JOINTS),
        d_embedding=1024,
        dropout=0.1,
        rendering_resolution=256,
        target_resolution=512,
        edges=EDGES,
        widths=WIDTHS,
        appearance_features=16,
    )

    input = RecursiveMunch.fromDict(
        {
            "A": {
                "image": torch.randn(1, 3, 256, 256),
                "pose_2d": {
                    "p": torch.randn(1, 117, 2, 1),
                    "c": torch.randn(1, 117, 1, 1),
                },
                "pose_3d": {
                    "root": {"p": torch.randn(1, 1, 3, 1)},
                },
                "R": torch.randn(1, 3, 3),
                "t": torch.randn(1, 3, 1),
                "resolution": torch.randn(1, 2),
            },
            "B": {
                "R": torch.randn(1, 3, 3),
                "t": torch.randn(1, 3, 1),
                "K": torch.randn(1, 3, 3),
                "dist_coef": torch.randn(1, 5),
            },
        }
    )

    output = m(input)

    pprint(summarize_dict(output))
