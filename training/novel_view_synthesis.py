import argparse
import json
import os
from munch import Munch
from pprint import pprint
from pathlib import Path

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPShardedPlugin

from utils import min_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    args = parser.parse_args()

    experiment_dir = args.hparams.parent

    with args.hparams.open() as file:
        hparams = Munch.fromDict(json.load(file))

    with args.config.open() as file:
        config = Munch.fromDict(json.load(file))

    seed_everything(config.seed)

    config.num_cpus = os.cpu_count()
    config.num_gpus = torch.cuda.device_count()
    config.accumulate_grad_batches = min_max(
        x=hparams.batch_size // (config.num_gpus * config.max_batch_size_per_gpu),
        m=1,
    )
    assert config.num_gpus * config.max_batch_size_per_gpu * config.accumulate_grad_batches == hparams.batch_size
    config.num_workers = min_max(x=(config.num_cpus - 1) // config.num_gpus, m=1, M=4)
    config.strategy = DDPShardedPlugin() if config.num_gpus > 1 else None

    torch.set_num_threads(config.num_cpus)

    pprint(hparams.toDict())
    pprint(config.toDict())

    if "Panoptic" in config.root:
        from data.Panoptic.datamodule import PanopticImagePairDataModule
        from data.Panoptic.metadata import (
            VIEWS,
            KNOWN_SEQUENCES,
            TRAIN_KNOWN_INTERVALS,
            VAL_KNOWN_INTERVALS,
            TEST_KNOWN_INTERVALS,
        )

        datamodule = PanopticImagePairDataModule(
            root=config.root,
            sequences=config.sequences if config.sequences else KNOWN_SEQUENCES,
            train_intervals=config.train_intervals if config.train_intervals else TRAIN_KNOWN_INTERVALS,
            val_intervals=config.val_intervals if config.val_intervals else VAL_KNOWN_INTERVALS,
            test_intervals=config.test_intervals if config.test_intervals else TEST_KNOWN_INTERVALS,
            in_views=config.in_views if config.in_views else VIEWS,
            out_views=config.out_views if config.out_views else VIEWS,
            cr_size_A=1080,
            cr_size_B=1080,
            re_size_A=hparams.input_resolution,
            re_size_B=hparams.target_resolution,
            batch_size=config.max_batch_size_per_gpu,
            num_workers=config.num_workers,
        )

        from lightning.novel_view_synthesis import (
            LitPanopticPixelNovelViewSynthesis,
            LitPanopticVGGNovelViewSynthesis,
            LitPanopticLPIPSNovelViewSynthesis,
            LitPanopticAdversarialNovelViewSynthesis,
            LitPanopticAdaptiveAdversarialNovelViewSynthesis,
        )

        if hparams.image_loss == "pixel":
            litmodel = LitPanopticPixelNovelViewSynthesis
        elif hparams.image_loss == "vgg":
            litmodel = LitPanopticVGGNovelViewSynthesis
        elif hparams.image_loss == "lpips":
            litmodel = LitPanopticLPIPSNovelViewSynthesis
        elif hparams.image_loss == "adversarial":
            litmodel = LitPanopticAdversarialNovelViewSynthesis
        elif hparams.image_loss == "adaptive":
            litmodel = LitPanopticAdaptiveAdversarialNovelViewSynthesis
        else:
            raise ValueError(f"Unknown image_loss: {hparams.image_loss}")
    elif "Human3.6M" in config.root:
        from data.Human36M.datamodule import Human36MImagePairDataModule
        from data.Human36M.metadata import (
            VIEWS,
            KNOWN_SEQUENCES,
            TRAIN_KNOWN_INTERVALS,
            VAL_KNOWN_INTERVALS,
            TEST_KNOWN_INTERVALS,
        )

        datamodule = Human36MImagePairDataModule(
            root=config.root,
            sequences=config.sequences if config.sequences else KNOWN_SEQUENCES,
            train_intervals=config.train_intervals if config.train_intervals else TRAIN_KNOWN_INTERVALS,
            val_intervals=config.val_intervals if config.val_intervals else VAL_KNOWN_INTERVALS,
            test_intervals=config.test_intervals if config.test_intervals else TEST_KNOWN_INTERVALS,
            in_views=config.in_views if config.in_views else VIEWS,
            out_views=config.out_views if config.out_views else VIEWS,
            cr_margin_A=0.25,
            cr_margin_B=0.25,
            re_size_A=hparams.input_resolution,
            re_size_B=hparams.target_resolution,
            batch_size=config.max_batch_size_per_gpu,
            num_workers=config.num_workers,
        )

        from lightning.novel_view_synthesis import (
            LitHuman36MPixelNovelViewSynthesis,
            LitHuman36MVGGNovelViewSynthesis,
            LitHuman36MLPIPSNovelViewSynthesis,
            LitHuman36MAdversarialNovelViewSynthesis,
            LitHuman36MAdaptiveAdversarialNovelViewSynthesis,
        )

        if hparams.image_loss == "pixel":
            litmodel = LitHuman36MPixelNovelViewSynthesis
        elif hparams.image_loss == "vgg":
            litmodel = LitHuman36MVGGNovelViewSynthesis
        elif hparams.image_loss == "lpips":
            litmodel = LitHuman36MLPIPSNovelViewSynthesis
        elif hparams.image_loss == "adversarial":
            litmodel = LitHuman36MAdversarialNovelViewSynthesis
        elif hparams.image_loss == "adaptive":
            litmodel = LitHuman36MAdaptiveAdversarialNovelViewSynthesis
        else:
            raise ValueError(f"Unknown image_loss: {hparams.image_loss}")
    else:
        raise ValueError(f"Unknown root: {config.root}")

    if not args.ckpt:
        litmodel = litmodel(hparams=hparams.toDict())
    else:
        litmodel = litmodel.load_from_checkpoint(checkpoint_path=args.ckpt)

    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        verbose=True,
        save_last=True,
        save_top_k=1,
        mode="min",
    )
    learning_rate_monitor = LearningRateMonitor()
    tqdm_progress_bar = TQDMProgressBar(
        refresh_rate=64 * config.accumulate_grad_batches,
    )
    logger = TensorBoardLogger(
        save_dir=f"{experiment_dir}",
        name="lightning_logs",
        version=0,
        default_hp_metric=False,
    )

    checkpoints = sorted((experiment_dir / "lightning_logs").rglob("*/last.ckpt"))
    if checkpoints:
        checkpoint = str(checkpoints[-1])
    else:
        checkpoint = None

    trainer = Trainer(
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[
            model_checkpoint_callback,
            learning_rate_monitor,
            tqdm_progress_bar,
        ],
        default_root_dir=f"{experiment_dir}",
        logger=logger,
        gradient_clip_val=1.0,
        gpus=config.num_gpus,
        auto_select_gpus=True,
        benchmark=True,
        num_sanity_val_steps=0,
        strategy=config.strategy,
        precision=config.precision,
        max_epochs=config.max_epochs,
        limit_train_batches=config.limit_train_batches * config.accumulate_grad_batches,
        limit_val_batches=config.limit_val_batches * config.accumulate_grad_batches,
        flush_logs_every_n_steps=64 * config.accumulate_grad_batches,
        log_every_n_steps=32 * config.accumulate_grad_batches,
        resume_from_checkpoint=checkpoint,
        reload_dataloaders_every_n_epochs=1,
        detect_anomaly=True,
    )
    trainer.fit(model=litmodel, datamodule=datamodule)
    trainer.test(model=litmodel, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
