from typing import Any, Dict

from munch import RecursiveMunch

import torch
from torch import Tensor
from torch.optim import AdamW
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from torchvision.utils import make_grid

from lightning.base import LitModule

from models.pose_2d_to_pose_3d import Pose2DToPose3D

from operations.losses.pose import MPJPELoss, MPVJPELoss
from operations.metrics.pose import (
    MPJPEMetric,
    NMPJPEMetric,
    PMPJPEMetric,
    MPVJPEMetric,
)

from geometry.vector import split
from geometry.intrinsic import perspective_projection, rescale_pixels

from utils.plot import draw_limbs


class LitPose2DToPose3D(LitModule):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitPose2DToPose3D, self).__init__(hparams=hparams)

        self.model = Pose2DToPose3D(
            n_joints=len(self.hparams.joints),
            d_embedding=self.hparams.d_embedding,
            dropout=self.hparams.dropout,
        )

    def forward(self, input):
        output = RecursiveMunch()
        output.pose_3d.relative.p = self.model(
            p=input.pose_2d.p,
            c=input.pose_2d.c,
            resolution=input.resolution,
        )
        return output

    def configure_optimizers(self):
        opt = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            amsgrad=True,
        )
        sch = CosineAnnealingWarmupRestarts(
            optimizer=opt,
            first_cycle_steps=200,
            max_lr=self.hparams.lr,
            min_lr=self.hparams.lr * 5e-2,
            warmup_steps=1,
        )
        return {"optimizer": opt, "lr_scheduler": sch}

    @torch.no_grad()
    def _add_epoch_summary(self, mode: str):
        tag = lambda x: f"{mode}/{x}"
        move = lambda x: x.cpu().clone()

        n = self.input.pose_2d.p.shape[0]
        k = min(n, 4)

        input = RecursiveMunch()
        output = RecursiveMunch()

        input.pose_2d.p = move(self.input.pose_2d.p[:k])
        input.pose_2d.c = move(self.input.pose_2d.c[:k])

        input.pose_3d.root.p = move(self.input.pose_3d.root.p[:k])
        input.pose_3d.root.c = move(self.input.pose_3d.root.c[:k])

        input.pose_3d.relative.p = move(self.input.pose_3d.relative.p[:k])
        input.pose_3d.relative.c = move(self.input.pose_3d.relative.c[:k])

        input.pose_3d.p = input.pose_3d.root.p + input.pose_3d.relative.p
        input.pose_3d.c = input.pose_3d.root.c & input.pose_3d.relative.c

        output.pose_3d.relative.p = move(self.output.pose_3d.relative.p[:k])

        output.pose_3d.p = input.pose_3d.root.p + output.pose_3d.relative.p
        output.pose_3d.c = torch.ones_like(input.pose_3d.c)

        input.K = move(self.input.K[:k])
        input.dist_coef = move(self.input.dist_coef[:k])
        input.resolution = move(self.input.resolution[:k])

        old_resolution = input.resolution
        w, h = self.summary_resolution
        new_resolution = torch.tensor([[w, h]], device=old_resolution.device)

        left = draw_limbs(
            image=torch.ones(k, 3, h, w),
            points=rescale_pixels(
                xy=input.pose_2d.p,
                old_resolution=old_resolution[..., None, :],
                new_resolution=new_resolution[..., None, :],
            ),
            confidences=input.pose_2d.c,
            edges=self.hparams.edges,
        )
        middle = draw_limbs(
            image=torch.ones(k, 3, h, w),
            points=rescale_pixels(
                xy=perspective_projection(
                    xyz=output.pose_3d.p,
                    K=input.K[..., None, :, :],
                    dist_coef=input.dist_coef[..., None, :],
                ),
                old_resolution=old_resolution[..., None, :],
                new_resolution=new_resolution[..., None, :],
            ),
            confidences=output.pose_3d.c,
            edges=self.hparams.edges,
        )
        right = draw_limbs(
            image=torch.ones(k, 3, h, w),
            points=rescale_pixels(
                xy=perspective_projection(
                    xyz=input.pose_3d.p,
                    K=input.K[..., None, :, :],
                    dist_coef=input.dist_coef[..., None, :],
                ),
                old_resolution=old_resolution[..., None, :],
                new_resolution=new_resolution[..., None, :],
            ),
            confidences=input.pose_3d.c,
            edges=self.hparams.edges,
        )
        mosaic = torch.cat([left, middle, right], dim=-1)

        self.logger.experiment.add_image(
            tag=tag("mosaic"),
            img_tensor=make_grid(mosaic, nrow=2),
            global_step=self.current_epoch,
        )
        self.logger.experiment.flush()


class LitHuman36MPose2DTo3D(LitPose2DToPose3D):
    def __init__(self, hparams: Dict[str, Any]):
        from data.Human36M.skeleton import JOINTS, EDGES
        
        hparams.update(joints=JOINTS, edges=EDGES)
        super(LitHuman36MPose2DTo3D, self).__init__(hparams=hparams)

        self.mpjpe_loss = MPJPELoss()

        self.mpjpe_metric = MPJPEMetric()
        self.nmpjpe_metric = NMPJPEMetric()
        self.pmpjpe_metric = PMPJPEMetric()

        self.n_body = 17

        self.root_index = self.hparams.joints["HipCenter"]

        self.summary_resolution = (256, 256)

    def _training_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        losses = RecursiveMunch()

        losses.relative = self.mpjpe_loss(
            input=output.pose_3d.relative.p,
            target=input.pose_3d.relative.p,
        )

        train_loss = losses.relative

        return losses, train_loss

    def _evaluation_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        losses = RecursiveMunch()

        losses.mpjpe = self.mpjpe_metric(
            input=output.pose_3d.relative.p,
            target=input.pose_3d.relative.p,
        )
        losses.nmpjpe = self.nmpjpe_metric(
            input=output.pose_3d.relative.p,
            target=input.pose_3d.relative.p,
        )
        losses.pmpjpe = self.pmpjpe_metric(
            input=output.pose_3d.relative.p,
            target=input.pose_3d.relative.p,
        )

        val_loss = losses.mpjpe

        return losses, val_loss


class LitPanopticPose2DTo3D(LitPose2DToPose3D):
    def __init__(self, hparams: Dict[str, Any]):
        from data.Panoptic.skeleton import JOINTS, EDGES

        hparams.update(joints=JOINTS, edges=EDGES)
        super(LitPanopticPose2DTo3D, self).__init__(hparams=hparams)

        self.mpvjpe_loss = MPVJPELoss()

        self.mpvjpe_metric = MPVJPEMetric()

        self.n_body = 19
        self.n_left_hand = 20
        self.n_right_hand = 20
        self.n_face = 58

        self.root_index = self.hparams.joints["Sternum"]
        self.left_wrist_index = self.hparams.joints["LWrist"]
        self.right_wrist_index = self.hparams.joints["RWrist"]
        self.nose_index = self.hparams.joints["Nose"]

        self.summary_resolution = (480, 270)

    def split_p(self, p: Tensor):
        pb, plh, prh, pf = p.split(
            [self.n_body, self.n_left_hand, self.n_right_hand, self.n_face],
            dim=-3,
        )

        plw, _ = split(pb, index=self.left_wrist_index, dim=-3)
        prw, _ = split(pb, index=self.right_wrist_index, dim=-3)
        pn, _ = split(pb, index=self.nose_index, dim=-3)

        plh = plh - plw
        prh = prh - prw
        pf = pf - pn

        return pb, plh, prh, pf

    def split_c(self, c: Tensor):
        cb, clh, crh, cf = c.split(
            [self.n_body, self.n_left_hand, self.n_right_hand, self.n_face],
            dim=-3,
        )

        clw, _ = split(cb, index=self.left_wrist_index, dim=-3)
        crw, _ = split(cb, index=self.right_wrist_index, dim=-3)
        cn, _ = split(cb, index=self.nose_index, dim=-3)

        clh = clh & clw
        crh = crh & crw
        cf = cf & cn

        return cb, clh, crh, cf

    def _training_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        opb, oplh, oprh, opf = self.split_p(output.pose_3d.relative.p)
        ipb, iplh, iprh, ipf = self.split_p(input.pose_3d.relative.p)
        icb, iclh, icrh, icf = self.split_c(input.pose_3d.relative.c)

        losses = RecursiveMunch()

        losses.body = self.mpvjpe_loss(input=opb, target=ipb, weight=icb)
        losses.left_hand = self.mpvjpe_loss(input=oplh, target=iplh, weight=iclh)
        losses.right_hand = self.mpvjpe_loss(input=oprh, target=iprh, weight=icrh)
        losses.face = self.mpvjpe_loss(input=opf, target=ipf, weight=icf)

        train_loss = (
            1e0 * losses.body
            + 1e1 * losses.left_hand
            + 1e1 * losses.right_hand
            + 5e0 * losses.face
        )

        return losses, train_loss

    def _evaluation_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        opb, oplh, oprh, opf = self.split_p(output.pose_3d.relative.p)
        ipb, iplh, iprh, ipf = self.split_p(input.pose_3d.relative.p)
        icb, iclh, icrh, icf = self.split_c(input.pose_3d.relative.c)

        losses = RecursiveMunch()

        losses.body = self.mpvjpe_metric(input=opb, target=ipb, weight=icb)
        losses.left_hand = self.mpvjpe_metric(input=oplh, target=iplh, weight=iclh)
        losses.right_hand = self.mpvjpe_metric(input=oprh, target=iprh, weight=icrh)
        losses.face = self.mpvjpe_metric(input=opf, target=ipf, weight=icf)

        val_loss = losses.body

        return losses, val_loss
