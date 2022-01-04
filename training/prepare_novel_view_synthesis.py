import argparse
import json
from munch import Munch
from pathlib import Path

import torch

import pytorch_lightning
from pytorch_lightning import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_pose_2d_to_pose_3d", type=Path, required=True)
    parser.add_argument("--ckpt_pose_2d_to_pose_3d", type=Path, required=True)
    parser.add_argument("--hparams_novel_view_synthesis", type=Path, required=True)
    parser.add_argument("--ckpt_novel_view_synthesis", type=Path, required=True)
    args = parser.parse_args()

    with args.hparams_novel_view_synthesis.open() as file:
        hparams_novel_view_synthesis = Munch.fromDict(json.load(file))

    with args.config_pose_2d_to_pose_3d.open() as file:
        config_pose_2d_to_pose_3d = Munch.fromDict(json.load(file))

    seed_everything(config_pose_2d_to_pose_3d.seed)

    if "Panoptic" in config_pose_2d_to_pose_3d.root:
        from lightning.pose_2d_to_pose_3d import LitPanopticPose2DTo3D

        lit_pose_2d_to_pose_3d = LitPanopticPose2DTo3D

        from lightning.novel_view_synthesis import (
            LitPanopticPixelNovelViewSynthesis,
            LitPanopticVGGNovelViewSynthesis,
            LitPanopticLPIPSNovelViewSynthesis,
            LitPanopticAdversarialNovelViewSynthesis,
            LitPanopticAdaptiveAdversarialNovelViewSynthesis,
        )

        if hparams_novel_view_synthesis.image_loss == "pixel":
            lit_novel_view_synthesis = LitPanopticPixelNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "vgg":
            lit_novel_view_synthesis = LitPanopticVGGNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "lpips":
            lit_novel_view_synthesis = LitPanopticLPIPSNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "adversarial":
            lit_novel_view_synthesis = LitPanopticAdversarialNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "adaptive":
            lit_novel_view_synthesis = LitPanopticAdaptiveAdversarialNovelViewSynthesis
        else:
            raise ValueError(f"Unknown image_loss: {hparams_novel_view_synthesis.image_loss}")
    elif "Human3.6M" in config_pose_2d_to_pose_3d.root:
        from lightning.pose_2d_to_pose_3d import LitHuman36MPose2DTo3D

        lit_pose_2d_to_pose_3d = LitHuman36MPose2DTo3D

        from lightning.novel_view_synthesis import (
            LitHuman36MPixelNovelViewSynthesis,
            LitHuman36MVGGNovelViewSynthesis,
            LitHuman36MLPIPSNovelViewSynthesis,
            LitHuman36MAdversarialNovelViewSynthesis,
            LitHuman36MAdaptiveAdversarialNovelViewSynthesis,
        )

        if hparams_novel_view_synthesis.image_loss == "pixel":
            lit_novel_view_synthesis = LitHuman36MPixelNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "vgg":
            lit_novel_view_synthesis = LitHuman36MVGGNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "lpips":
            lit_novel_view_synthesis = LitHuman36MLPIPSNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "adversarial":
            lit_novel_view_synthesis = LitHuman36MAdversarialNovelViewSynthesis
        elif hparams_novel_view_synthesis.image_loss == "adaptive":
            lit_novel_view_synthesis = LitHuman36MAdaptiveAdversarialNovelViewSynthesis
        else:
            raise ValueError(f"Unknown image_loss: {hparams_novel_view_synthesis.image_loss}")
    else:
        raise ValueError(f"Unknown root: {config_pose_2d_to_pose_3d.root}")

    lit_pose_2d_to_pose_3d = lit_pose_2d_to_pose_3d.load_from_checkpoint(checkpoint_path=args.ckpt_pose_2d_to_pose_3d)

    lit_novel_view_synthesis = lit_novel_view_synthesis(hparams=hparams_novel_view_synthesis.toDict())

    lit_novel_view_synthesis.model.p2p.load_state_dict(lit_pose_2d_to_pose_3d.model.state_dict())

    checkpoint = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": pytorch_lightning.__version__,
        "state_dict": lit_novel_view_synthesis.state_dict(),
        "hparams_name": "hparams",
        "hyper_parameters": dict(lit_novel_view_synthesis.hparams),
    }

    args.ckpt_novel_view_synthesis.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.ckpt_novel_view_synthesis)


if __name__ == '__main__':
    main()
