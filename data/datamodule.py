from typing import Optional, Union

from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super(DataModule, self).__init__()

        if not isinstance(root, Path):
            root = Path(root)

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._train_dataset()
            self.val_dataset = self._val_dataset()

        if stage == "test" or stage is None:
            self.test_dataset = self._test_dataset()

    def _train_dataset(self):
        raise NotImplementedError

    def _val_dataset(self):
        raise NotImplementedError

    def _test_dataset(self):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
