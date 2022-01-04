from typing import Iterable, Union, Tuple

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import ConcatDataset
from data.datamodule import DataModule
from data.utils import filter

from data.Human36M.dataset import (
    Human36MPoseDataset,
    Human36MImageDataset,
    Human36MImagePairDataset,
)
from data.Human36M.transform import (
    Human36MImageTransform,
    Human36MImagePairTransform,
)
from data.Human36M.metadata import (
    VIEWS,
    KNOWN_SEQUENCES,
    TRAIN_KNOWN_INTERVALS,
    VAL_KNOWN_INTERVALS,
    TEST_KNOWN_INTERVALS,
    UNKNOWN_SEQUENCES,
    TRAIN_UNKNOWN_INTERVALS,
    VAL_UNKNOWN_INTERVALS,
    TEST_UNKNOWN_INTERVALS,
)


class Human36MPoseDataModule(DataModule):
    def __init__(
        self,
        root: Union[str, Path],
        known_sequences: Iterable[str] = KNOWN_SEQUENCES,
        train_intervals: Iterable = TRAIN_KNOWN_INTERVALS,
        unknown_sequences: Iterable[str] = UNKNOWN_SEQUENCES,
        views: Iterable[str] = VIEWS,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super(Human36MPoseDataModule, self).__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.known_sequences = known_sequences
        self.train_intervals = train_intervals
        self.unknown_sequences = unknown_sequences
        self.views = views

    def _train_dataset(self):
        datasets = []
        for sequence, intervals in zip(
            tqdm(self.known_sequences), self.train_intervals
        ):
            for view in tqdm(self.views, leave=False):
                dataset = Human36MPoseDataset(
                    path=self.root / sequence,
                    view=view,
                )
                dataset.keys = filter(dataset.keys, intervals)
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset

    def _val_dataset(self):
        datasets = []
        for sequence in tqdm(self.unknown_sequences):
            for view in tqdm(self.views, leave=False):
                dataset = Human36MPoseDataset(
                    path=self.root / sequence,
                    view=view,
                )
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset

    def _test_dataset(self):
        datasets = []
        for sequence in tqdm(self.unknown_sequences):
            for view in tqdm(self.views, leave=False):
                dataset = Human36MPoseDataset(
                    path=self.root / sequence,
                    view=view,
                )
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset


class Human36MImageDataModule(DataModule):
    def __init__(
        self,
        root: Union[str, Path],
        sequences: Iterable[str] = KNOWN_SEQUENCES,
        train_intervals: Iterable = TRAIN_KNOWN_INTERVALS,
        val_intervals: Iterable = VAL_KNOWN_INTERVALS,
        test_intervals: Iterable = TEST_KNOWN_INTERVALS,
        views: Iterable[str] = VIEWS,
        cr_margin: Union[float, Tuple[float, float]] = None,
        re_size: Union[int, Tuple[int, int]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super(Human36MImageDataModule, self).__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.sequences = sequences
        self.train_intervals = train_intervals
        self.val_intervals = val_intervals
        self.test_intervals = test_intervals
        self.views = views

        self.transform = Human36MImageTransform(
            cr_margin=cr_margin,
            re_size=re_size,
        )

    def _train_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.train_intervals):
            for view in tqdm(self.views, leave=False):
                dataset = Human36MImageDataset(
                    path=self.root / sequence,
                    view=view,
                    transform=self.transform,
                )
                dataset.keys = filter(dataset.keys, intervals)
                datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets

    def _val_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.val_intervals):
            for view in tqdm(self.views, leave=False):
                dataset = Human36MImageDataset(
                    path=self.root / sequence,
                    view=view,
                    transform=self.transform,
                )
                dataset.keys = filter(dataset.keys, intervals)
                datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets

    def _test_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.test_intervals):
            for view in tqdm(self.views, leave=False):
                dataset = Human36MImageDataset(
                    path=self.root / sequence,
                    view=view,
                    transform=self.transform,
                )
                dataset.keys = filter(dataset.keys, intervals)
                datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets


class Human36MImagePairDataModule(DataModule):
    def __init__(
        self,
        root: Union[str, Path],
        sequences: Iterable[str] = KNOWN_SEQUENCES,
        train_intervals: Iterable = TRAIN_KNOWN_INTERVALS,
        val_intervals: Iterable = VAL_KNOWN_INTERVALS,
        test_intervals: Iterable = TEST_KNOWN_INTERVALS,
        in_views: Iterable[str] = VIEWS,
        out_views: Iterable[str] = VIEWS,
        cr_margin_A: Union[float, Tuple[float, float]] = None,
        cr_margin_B: Union[float, Tuple[float, float]] = None,
        re_size_A: Union[int, Tuple[int, int]] = None,
        re_size_B: Union[int, Tuple[int, int]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super(Human36MImagePairDataModule, self).__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.sequences = sequences
        self.train_intervals = train_intervals
        self.val_intervals = val_intervals
        self.test_intervals = test_intervals
        self.in_views = in_views
        self.out_views = out_views

        self.transform = Human36MImagePairTransform(
            cr_margin_A=cr_margin_A,
            cr_margin_B=cr_margin_B,
            re_size_A=re_size_A,
            re_size_B=re_size_B,
        )

    def _train_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.train_intervals):
            dataset = Human36MImagePairDataset(
                path=self.root / sequence,
                in_views=self.in_views,
                out_views=self.out_views,
                transform=self.transform,
            )
            dataset.keys = filter(dataset.keys, intervals)
            datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets

    def _val_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.val_intervals):
            dataset = Human36MImagePairDataset(
                path=self.root / sequence,
                in_views=self.in_views,
                out_views=self.out_views,
                transform=self.transform,
            )
            dataset.keys = filter(dataset.keys, intervals)
            datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets

    def _test_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.test_intervals):
            dataset = Human36MImagePairDataset(
                path=self.root / sequence,
                in_views=self.in_views,
                out_views=self.out_views,
                transform=self.transform,
            )
            dataset.keys = filter(dataset.keys, intervals)
            datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets
