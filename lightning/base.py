from typing import Any, Dict

import torch
from torch import Tensor

from pytorch_lightning import LightningModule

from utils.logging import git_hash, flatten_dict


class LitModule(LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitModule, self).__init__()
        hparams["hash"] = git_hash()

        self.save_hyperparameters(hparams)

        self.input = None
        self.output = None

    def training_step(self, input, batch_idx) -> Tensor:
        losses, train_loss = self._training_step(input=input, batch_idx=batch_idx)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._add_epoch_summary(mode="train")

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            self.log_dict(
                dictionary={f"train/{k}": v for k, v in flatten_dict(losses).items()},
                sync_dist=False,
            )

        return train_loss

    def validation_step(self, input, batch_idx):
        losses, val_loss = self._evaluation_step(input=input, batch_idx=batch_idx)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._add_epoch_summary(mode="val")

        self.log(f"val_loss", val_loss, sync_dist=True)
        self.log_dict(
            dictionary={f"val/{k}": v for k, v in flatten_dict(losses).items()},
            sync_dist=True,
        )

    def test_step(self, input, batch_idx):
        losses, _ = self._evaluation_step(input=input, batch_idx=batch_idx)

        self.log_dict(
            dictionary={f"test/{k}": v for k, v in flatten_dict(losses).items()},
            sync_dist=True,
        )

    def _training_step(self, input, batch_idx):
        raise NotImplementedError

    def _evaluation_step(self, input, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def on_train_start(self):
        self.logger.log_hyperparams(
            params=self.hparams,
            metrics={
                "val_loss": 0.0,
            },
        )

    def on_epoch_start(self):
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def on_epoch_end(self):
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _add_epoch_summary(self, mode: str):
        raise NotImplementedError
