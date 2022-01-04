from typing import Any, Dict

from munch import Munch, RecursiveMunch

import torch
from torch import Tensor
from torch.nn import L1Loss, MSELoss
from torch.optim import AdamW

from torchvision.utils import make_grid

from lightning.base import LitModule

from models.novel_view_synthesis import NovelViewSynthesis
from models.discriminator import Discriminator
from nn.augmentations import AdaptiveAugmentations

from operations.losses.pose import MPJPELoss, MPVJPELoss
from operations.losses.image import LPIPSLoss, VGGLoss
from operations.losses.adversarial import AdversarialLoss, R1, AdversarialFeaturesLoss
from operations.metrics.image import PSNRMetric, SSIMMetric, LPIPSMetric
from operations.metrics.pose import (
    MPJPEMetric,
    NMPJPEMetric,
    PMPJPEMetric,
    MPVJPEMetric,
)

from geometry.intrinsic import perspective_projection
from geometry.vector import split

from utils.plot import draw_limbs
from utils.logging import git_hash, flatten_dict


class LitNovelViewSynthesis(LitModule):
    def __init__(self, hparams: Dict[str, Any]):
        hparams.update(hash=git_hash())
        super(LitNovelViewSynthesis, self).__init__(hparams=hparams)

        self.model = NovelViewSynthesis(
            n_joints=len(self.hparams.joints),
            d_embedding=self.hparams.d_embedding,
            dropout=self.hparams.dropout,
            rendering_resolution=self.hparams.rendering_resolution,
            target_resolution=self.hparams.target_resolution,
            edges=self.hparams.edges,
            widths=self.hparams.widths,
            appearance_features=self.hparams.appearance_features,
            base_channels=self.hparams.base_channels,
        )

        self.psnr_metric = PSNRMetric()
        self.ssim_metric = SSIMMetric()
        self.lpips_metric = LPIPSMetric()

    def forward(self, input: Munch) -> Munch:
        output = self.model(input)
        return output

    def _training_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        image_losses, image_train_loss = self._training_step_image(input, output)
        pose_losses, pose_train_loss = self._training_step_pose(input, output)

        losses = RecursiveMunch()
        losses.update(image_losses)
        losses.update(pose_losses)

        train_loss = image_train_loss + pose_train_loss

        return losses, train_loss

    def _training_step_image(self, input, output):
        raise NotImplementedError

    def _training_step_pose(self, input, output):
        raise NotImplementedError

    def _evaluation_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        image_losses = self._evaluation_step_image(input, output)
        pose_losses = self._evaluation_step_pose(input, output)

        losses = RecursiveMunch()
        losses.update(image_losses)
        losses.update(pose_losses)

        val_loss = losses.lpips

        return losses, val_loss

    def _evaluation_step_image(self, input, output):
        losses = RecursiveMunch()

        losses.psnr = self.psnr_metric(
            input=output.B.image.clamp(min=0.0, max=1.0),
            target=input.B.masked_image,
        )
        losses.ssim = self.ssim_metric(
            input=output.B.image.clamp(min=0.0, max=1.0),
            target=input.B.masked_image,
        )
        losses.lpips = self.lpips_metric(
            input=output.B.image.clamp(min=0.0, max=1.0),
            target=input.B.masked_image,
        )

        return losses

    def _evaluation_step_pose(self, input, output):
        raise NotImplementedError

    def configure_optimizers(self):
        opt = AdamW(
            params=[
                {"params": self.model.p2p.parameters(), "lr": 2.5e-3 * self.hparams.lr},
                {"params": self.model.i2a.parameters(), "lr": 1e-1 * self.hparams.lr},
                {"params": self.model.r2i.parameters(), "lr": self.hparams.lr},
            ],
            betas=(0.0, 0.99),
            weight_decay=self.hparams.wd,
            amsgrad=True,
        )
        return opt

    @torch.no_grad()
    def _add_epoch_summary(self, mode: str):
        tag = lambda x: f"{mode}/{x}"

        def move(x: Tensor) -> Tensor:
            x = x.cpu()
            if x.dtype == torch.half:
                x = x.float()
            x = x.clone()
            return x

        n, c, h, w = self.input.A.image.shape
        k = min(n, 4)

        input = RecursiveMunch()
        output = RecursiveMunch()

        input.A.image = move(self.input.A.image[:k])
        input.B.image = move(self.input.B.image[:k])
        output.B.image = move(self.output.B.image[:k])
        input.B.masked_image = move(self.input.B.masked_image[:k])

        input.A.pose_2d.p = move(self.input.A.pose_2d.p[:k])
        input.A.pose_2d.c = move(self.input.A.pose_2d.c[:k])

        input.A.pose_3d.root.p = move(self.input.A.pose_3d.root.p[:k])
        input.A.pose_3d.root.c = move(self.input.A.pose_3d.root.c[:k])

        input.A.pose_3d.relative.p = move(self.input.A.pose_3d.relative.p[:k])
        input.A.pose_3d.relative.c = move(self.input.A.pose_3d.relative.c[:k])

        input.A.pose_3d.p = input.A.pose_3d.root.p + input.A.pose_3d.relative.p
        input.A.pose_3d.c = input.A.pose_3d.root.c & input.A.pose_3d.relative.c

        output.A.pose_3d.relative.p = move(self.output.A.pose_3d.relative.p[:k])

        output.A.pose_3d.p = input.A.pose_3d.root.p + output.A.pose_3d.relative.p
        output.A.pose_3d.c = torch.ones_like(input.A.pose_3d.c)

        input.B.pose_3d.root.p = move(self.input.B.pose_3d.root.p[:k])
        input.B.pose_3d.root.c = move(self.input.B.pose_3d.root.c[:k])

        input.B.pose_3d.relative.p = move(self.input.B.pose_3d.relative.p[:k])
        input.B.pose_3d.relative.c = move(self.input.B.pose_3d.relative.c[:k])

        input.B.pose_3d.p = input.B.pose_3d.root.p + input.B.pose_3d.relative.p
        input.B.pose_3d.c = input.B.pose_3d.root.c & input.B.pose_3d.relative.c

        output.B.pose_3d.p = move(self.output.B.pose_3d.p[:k])
        output.B.pose_3d.c = torch.ones_like(input.A.pose_3d.c)

        input.A.K = move(self.input.A.K[:k])
        input.A.dist_coef = move(self.input.A.dist_coef[:k])

        input.B.K = move(self.input.B.K[:k])
        input.B.dist_coef = move(self.input.B.dist_coef[:k])

        left = input.A.image
        middle = draw_limbs(
            image=input.A.image.clone(),
            points=perspective_projection(
                xyz=output.A.pose_3d.p,
                K=input.A.K[..., None, :, :],
                dist_coef=input.A.dist_coef[..., None, :],
            ),
            confidences=output.A.pose_3d.c,
            edges=self.hparams.edges,
        )
        right = draw_limbs(
            image=input.A.image.clone(),
            points=perspective_projection(
                xyz=input.A.pose_3d.p,
                K=input.A.K[..., None, :, :],
                dist_coef=input.A.dist_coef[..., None, :],
            ),
            confidences=input.A.pose_3d.c,
            edges=self.hparams.edges,
        )
        mosaic = torch.cat([left, middle, right], dim=-1)
        self.logger.experiment.add_image(
            tag=tag("input"),
            img_tensor=make_grid(mosaic, nrow=1),
            global_step=self.global_step,
        )

        left = output.B.image.clamp(min=0.0, max=1.0)
        middle_left = input.B.masked_image
        middle = input.B.image
        middle_right = draw_limbs(
            image=output.B.image.clamp(min=0.0, max=1.0).clone(),
            points=perspective_projection(
                xyz=output.B.pose_3d.p,
                K=input.B.K[..., None, :, :],
                dist_coef=input.B.dist_coef[..., None, :],
            ),
            confidences=output.B.pose_3d.c,
            edges=self.hparams.edges,
        )
        right = draw_limbs(
            image=input.B.masked_image.clone(),
            points=perspective_projection(
                xyz=input.B.pose_3d.p,
                K=input.B.K[..., None, :, :],
                dist_coef=input.B.dist_coef[..., None, :],
            ),
            confidences=input.B.pose_3d.c,
            edges=self.hparams.edges,
        )
        mosaic = torch.cat([left, middle_left, middle, middle_right, right], dim=-1)
        self.logger.experiment.add_image(
            tag=tag("output"),
            img_tensor=make_grid(mosaic, nrow=1),
            global_step=self.global_step,
        )

        self.logger.experiment.flush()


class LitPixelNovelViewSynthesis(LitNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitPixelNovelViewSynthesis, self).__init__(hparams=hparams)

        self.mae_loss = L1Loss()
        self.mse_loss = MSELoss()

    def _training_step_image(self, input, output):
        losses = RecursiveMunch()

        losses.psnr = self.psnr_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.ssim = self.ssim_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )

        losses.pixel = self.mae_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.appearance = self.mse_loss(
            input=output.A.appearance,
            target=torch.zeros_like(output.A.appearance),
        )

        train_loss = 1e1 * losses.pixel + 1e-3 * losses.appearance

        return losses, train_loss


class LitVGGNovelViewSynthesis(LitNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitVGGNovelViewSynthesis, self).__init__(hparams=hparams)

        self.mae_loss = L1Loss()
        self.mse_loss = MSELoss()
        self.vgg_loss = VGGLoss()

    def _training_step_image(self, input, output):
        losses = RecursiveMunch()

        losses.psnr = self.psnr_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.ssim = self.ssim_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )

        losses.pixel = self.mae_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.vgg = self.vgg_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.appearance = self.mse_loss(
            input=output.A.appearance,
            target=torch.zeros_like(output.A.appearance),
        )

        train_loss = 1e1 * losses.pixel + 1e1 * losses.vgg + 1e-3 * losses.appearance

        return losses, train_loss


class LitLPIPSNovelViewSynthesis(LitNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitLPIPSNovelViewSynthesis, self).__init__(hparams=hparams)

        self.mae_loss = L1Loss()
        self.mse_loss = MSELoss()
        self.lpips_loss = LPIPSLoss()

    def _training_step_image(self, input, output):
        losses = RecursiveMunch()

        losses.psnr = self.psnr_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.ssim = self.ssim_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )

        losses.pixel = self.mae_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.lpips = self.lpips_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.appearance = self.mse_loss(
            input=output.A.appearance,
            target=torch.zeros_like(output.A.appearance),
        )

        train_loss = 1e1 * losses.pixel + 1e1 * losses.lpips + 1e-3 * losses.appearance

        return losses, train_loss


class LitAdversarialNovelViewSynthesis(LitNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitAdversarialNovelViewSynthesis, self).__init__(hparams=hparams)

        self.mae_loss = L1Loss()
        self.mse_loss = MSELoss()
        self.lpips_loss = LPIPSLoss()

        self.discriminator = Discriminator()
        self.mbce_loss = AdversarialLoss()
        self.amae_loss = AdversarialFeaturesLoss()
        self.r1 = R1()

    def training_step(self, input, batch_idx, optimizer_idx) -> Tensor:
        if optimizer_idx == 0:
            losses, train_loss = self._generator_step(input, batch_idx)

            if batch_idx == 0 and self.trainer.is_global_zero:
                self._add_epoch_summary(mode="train")

        else:
            losses, train_loss = self._discriminator_step(input, batch_idx)

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.log_dict(
                dictionary={f"train/{k}": v for k, v in flatten_dict(losses).items()},
                sync_dist=False,
            )

        return train_loss

    def _generator_step(self, input, batch_idx):
        input = self.input = RecursiveMunch.fromDict(input)
        output = self.output = self(input)

        image_losses, image_train_loss = self._training_step_image(input, output)
        pose_losses, pose_train_loss = self._training_step_pose(input, output)

        losses = RecursiveMunch()
        losses.update(image_losses)
        losses.update(pose_losses)

        train_loss = image_train_loss + pose_train_loss

        return losses, train_loss

    def _training_step_image(self, input, output):
        losses = RecursiveMunch()

        losses.psnr = self.psnr_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.ssim = self.ssim_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )

        losses.pixel = self.mae_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.lpips = self.lpips_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.appearance = self.mse_loss(
            input=output.A.appearance,
            target=torch.zeros_like(output.A.appearance),
        )

        f_real, _ = self.discriminator(input.B.masked_image)
        f_fake, d_fake = self.discriminator(output.B.image)

        losses.generator = self.mbce_loss(input=d_fake, target=True)
        losses.adversarial_features = self.amae_loss(input=f_fake, target=f_real)

        train_loss = (
            1e1 * losses.pixel
            + 1e1 * losses.lpips
            + 1e-3 * losses.appearance
            + losses.generator
            + 1e1 * losses.adversarial_features
        )

        return losses, train_loss

    def _discriminator_step(self, input, batch_idx):
        input = self.input
        output = self.output

        input.B.masked_image = input.B.masked_image.detach().requires_grad_()
        output.B.image = output.B.image.detach()

        losses = RecursiveMunch()

        _, d_real = self.discriminator(input.B.masked_image)
        _, d_fake = self.discriminator(output.B.image)

        losses.discriminator.real = self.mbce_loss(input=d_real, target=True)
        losses.discriminator.fake = self.mbce_loss(input=d_fake, target=False)

        if (batch_idx // self.trainer.accumulate_grad_batches) % 16 == 0:
            losses.discriminator.r1 = self.r1(
                outputs=losses.discriminator.real,
                inputs=input.B.masked_image,
            )
        else:
            losses.discriminator.r1 = torch.zeros_like(losses.discriminator.real)

        h, w = input.B.masked_image.shape[-2:]
        gamma = 2.5e-4 * (h * w) / 32

        train_loss = (
            0.5 * (losses.discriminator.real + losses.discriminator.fake)
            + gamma * losses.discriminator.r1
        )

        return losses, train_loss

    def configure_optimizers(self):
        opt_m = AdamW(
            params=[
                # {"params": self.model.p2p.parameters(), "lr": 2.5e-3 * self.hparams.lr},
                {"params": self.model.i2a.parameters(), "lr": 1e-1 * self.hparams.lr},
                {"params": self.model.r2i.parameters(), "lr": self.hparams.lr},
            ],
            betas=(0.0, 0.99),
            weight_decay=self.hparams.wd,
            amsgrad=True,
        )
        opt_d = AdamW(
            params=self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(0.0, 0.99),
            weight_decay=self.hparams.wd,
        )
        return [opt_m, opt_d]


class LitAdaptiveAdversarialNovelViewSynthesis(LitAdversarialNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitAdaptiveAdversarialNovelViewSynthesis, self).__init__(hparams=hparams)

        self.adaptive_augmentations = AdaptiveAugmentations(p=0.0)

    def _training_step_image(self, input, output):
        losses = RecursiveMunch()

        losses.psnr = self.psnr_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.ssim = self.ssim_metric(
            input=output.B.image,
            target=input.B.masked_image,
        )

        losses.pixel = self.mae_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.lpips = self.lpips_loss(
            input=output.B.image,
            target=input.B.masked_image,
        )
        losses.appearance = self.mse_loss(
            input=output.A.appearance,
            target=torch.zeros_like(output.A.appearance),
        )

        theta = self.adaptive_augmentations.sample_theta(input=input.B.masked_image)
        phi = self.adaptive_augmentations.sample_phi(input=input.B.masked_image)

        input.B.augmented_image = self.adaptive_augmentations(
            input=input.B.masked_image,
            theta=theta,
            phi=phi,
        )
        output.B.augmented_image = self.adaptive_augmentations(
            input=output.B.image,
            theta=theta,
            phi=phi,
        )

        f_real, _ = self.discriminator(input.B.augmented_image)
        f_fake, d_fake = self.discriminator(output.B.augmented_image)

        losses.generator = self.mbce_loss(input=d_fake, target=True)
        losses.adversarial_features = self.amae_loss(input=f_fake, target=f_real)

        train_loss = (
            1e1 * losses.pixel
            + 1e1 * losses.lpips
            + 1e-3 * losses.appearance
            + losses.generator
            + 1e1 * losses.adversarial_features
        )

        return losses, train_loss

    def _discriminator_step(self, input, batch_idx):
        input = self.input
        output = self.output

        input.B.augmented_image = input.B.augmented_image.detach().requires_grad_()
        output.B.augmented_image = output.B.augmented_image.detach()

        losses = RecursiveMunch()

        _, d_real = self.discriminator(input.B.augmented_image)
        _, d_fake = self.discriminator(output.B.augmented_image)

        losses.discriminator.real = self.mbce_loss(input=d_real, target=True)
        losses.discriminator.fake = self.mbce_loss(input=d_fake, target=False)

        if (batch_idx // self.trainer.accumulate_grad_batches) % 16 == 0:
            losses.discriminator.r1 = self.r1(
                outputs=losses.discriminator.real,
                inputs=input.B.augmented_image,
            )
        else:
            losses.discriminator.r1 = torch.zeros_like(losses.discriminator.real)

        self.adaptive_augmentations.update(r_current=d_real.sign().mean())
        losses.adaptive_augmentations.probability = self.adaptive_augmentations.p
        losses.adaptive_augmentations.r_average = (
            self.adaptive_augmentations.ema.average
        )

        h, w = input.B.augmented_image.shape[-2:]
        gamma = 2.5e-4 * (h * w) / 32

        train_loss = (
            0.5 * (losses.discriminator.real + losses.discriminator.fake)
            + gamma * losses.discriminator.r1
        )

        return losses, train_loss


class LitHuman36MNovelViewSynthesis(LitNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        from data.Human36M.skeleton import JOINTS, EDGES
        from data.Human36M.statistics import WIDTHS

        hparams.update(joints=JOINTS, edges=EDGES, widths=WIDTHS)
        super(LitHuman36MNovelViewSynthesis, self).__init__(hparams=hparams)

        self.mpjpe_loss = MPJPELoss()

        self.mpjpe_metric = MPJPEMetric()
        self.nmpjpe_metric = NMPJPEMetric()
        self.pmpjpe_metric = PMPJPEMetric()

        self.n_body = 17

        self.root_index = self.hparams.joints["HipCenter"]

    def _training_step_pose(self, input, output):
        losses = RecursiveMunch()

        losses.relative = self.mpjpe_loss(
            input=output.A.pose_3d.relative.p,
            target=input.A.pose_3d.relative.p,
        )

        train_loss = losses.relative

        return losses, train_loss

    def _evaluation_step_pose(self, input, output):
        losses = RecursiveMunch()

        losses.mpjpe = self.mpjpe_metric(
            input=output.A.pose_3d.relative.p,
            target=input.A.pose_3d.relative.p,
        )
        losses.nmpjpe = self.nmpjpe_metric(
            input=output.A.pose_3d.relative.p,
            target=input.A.pose_3d.relative.p,
        )
        losses.pmpjpe = self.pmpjpe_metric(
            input=output.A.pose_3d.relative.p,
            target=input.A.pose_3d.relative.p,
        )

        return losses


class LitPanopticNovelViewSynthesis(LitNovelViewSynthesis):
    def __init__(self, hparams: Dict[str, Any]):
        from data.Panoptic.skeleton import JOINTS, EDGES
        from data.Panoptic.statistics import WIDTHS

        hparams.update(joints=JOINTS, edges=EDGES, widths=WIDTHS)
        super(LitPanopticNovelViewSynthesis, self).__init__(hparams=hparams)

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

    def _training_step_pose(self, input, output):
        losses = RecursiveMunch()

        opb, oplh, oprh, opf = self.split_p(output.A.pose_3d.relative.p)
        ipb, iplh, iprh, ipf = self.split_p(input.A.pose_3d.relative.p)
        icb, iclh, icrh, icf = self.split_c(input.A.pose_3d.relative.c)

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

    def _evaluation_step_pose(self, input, output):
        losses = RecursiveMunch()

        opb, oplh, oprh, opf = self.split_p(output.A.pose_3d.relative.p)
        ipb, iplh, iprh, ipf = self.split_p(input.A.pose_3d.relative.p)
        icb, iclh, icrh, icf = self.split_c(input.A.pose_3d.relative.c)

        losses.body = self.mpvjpe_metric(input=opb, target=ipb, weight=icb)
        losses.left_hand = self.mpvjpe_metric(input=oplh, target=iplh, weight=iclh)
        losses.right_hand = self.mpvjpe_metric(input=oprh, target=iprh, weight=icrh)
        losses.face = self.mpvjpe_metric(input=opf, target=ipf, weight=icf)

        return losses


class LitHuman36MPixelNovelViewSynthesis(
    LitHuman36MNovelViewSynthesis,
    LitPixelNovelViewSynthesis,
):
    pass


class LitHuman36MVGGNovelViewSynthesis(
    LitHuman36MNovelViewSynthesis,
    LitVGGNovelViewSynthesis,
):
    pass


class LitHuman36MLPIPSNovelViewSynthesis(
    LitHuman36MNovelViewSynthesis,
    LitLPIPSNovelViewSynthesis,
):
    pass


class LitHuman36MAdversarialNovelViewSynthesis(
    LitHuman36MNovelViewSynthesis,
    LitAdversarialNovelViewSynthesis,
):
    pass


class LitHuman36MAdaptiveAdversarialNovelViewSynthesis(
    LitHuman36MNovelViewSynthesis,
    LitAdaptiveAdversarialNovelViewSynthesis,
):
    pass


class LitPanopticPixelNovelViewSynthesis(
    LitPanopticNovelViewSynthesis,
    LitPixelNovelViewSynthesis,
):
    pass


class LitPanopticVGGNovelViewSynthesis(
    LitPanopticNovelViewSynthesis,
    LitVGGNovelViewSynthesis,
):
    pass


class LitPanopticLPIPSNovelViewSynthesis(
    LitPanopticNovelViewSynthesis,
    LitLPIPSNovelViewSynthesis,
):
    pass


class LitPanopticAdversarialNovelViewSynthesis(
    LitPanopticNovelViewSynthesis,
    LitAdversarialNovelViewSynthesis,
):
    pass


class LitPanopticAdaptiveAdversarialNovelViewSynthesis(
    LitPanopticNovelViewSynthesis,
    LitAdaptiveAdversarialNovelViewSynthesis,
):
    pass
