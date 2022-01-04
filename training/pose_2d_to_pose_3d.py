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
        from data.Panoptic.datamodule import PanopticPoseDataModule
        from data.Panoptic.metadata import (
            VIEWS,
            KNOWN_SEQUENCES,
            TRAIN_KNOWN_INTERVALS,
            UNKNOWN_SEQUENCES,
        )

        datamodule = PanopticPoseDataModule(
            root=config.root,
            known_sequences=config.known_sequences if config.known_sequences else KNOWN_SEQUENCES,
            train_intervals=config.train_intervals if config.train_intervals else TRAIN_KNOWN_INTERVALS,
            unknown_sequences=config.unknown_sequences if config.unknown_sequences else UNKNOWN_SEQUENCES,
            views=config.views if config.views else VIEWS,
            batch_size=config.max_batch_size_per_gpu,
            num_workers=config.num_workers,
        )

        from lightning.pose_2d_to_pose_3d import LitPanopticPose2DTo3D

        litmodel = LitPanopticPose2DTo3D
    elif "Human3.6M" in config.root:
        from data.Human36M.datamodule import Human36MPoseDataModule
        from data.Human36M.metadata import (
            VIEWS,
            KNOWN_SEQUENCES,
            TRAIN_KNOWN_INTERVALS,
            UNKNOWN_SEQUENCES,
        )

        datamodule = Human36MPoseDataModule(
            root=config.root,
            known_sequences=config.known_sequences if config.known_sequences else KNOWN_SEQUENCES,
            train_intervals=config.train_intervals if config.train_intervals else TRAIN_KNOWN_INTERVALS,
            unknown_sequences=config.unknown_sequences if config.unknown_sequences else UNKNOWN_SEQUENCES,
            views=config.views if config.views else VIEWS,
            batch_size=config.max_batch_size_per_gpu,
            num_workers=config.num_workers,
        )

        from lightning.pose_2d_to_pose_3d import LitHuman36MPose2DTo3D

        litmodel = LitHuman36MPose2DTo3D
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
        max_epochs=config.max_epochs,
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
