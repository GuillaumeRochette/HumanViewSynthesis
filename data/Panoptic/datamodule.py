from typing import Iterable, Union, Tuple

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import ConcatDataset
from data.datamodule import DataModule
from data.utils import filter

from data.Panoptic.dataset import (
    PanopticPoseDataset,
    PanopticImageDataset,
    PanopticImagePairDataset,
)
from data.Panoptic.transform import (
    PanopticImageTransform,
    PanopticImagePairTransform,
)
from data.Panoptic.metadata import (
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


class PanopticPoseDataModule(DataModule):
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
        super(PanopticPoseDataModule, self).__init__(
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
            dir = self.root / sequence / "Databases" / "Images"
            views = [p.stem for p in sorted(dir.glob("*.lmdb"))]
            views = [view for view in views if view in self.views]
            for view in tqdm(views, leave=False):
                dataset = PanopticPoseDataset(
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
            dir = self.root / sequence / "Databases" / "Images"
            views = [p.stem for p in sorted(dir.glob("*.lmdb"))]
            views = [view for view in views if view in self.views]
            for view in tqdm(views, leave=False):
                dataset = PanopticPoseDataset(
                    path=self.root / sequence,
                    view=view,
                )
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset

    def _test_dataset(self):
        datasets = []
        for sequence in tqdm(self.unknown_sequences):
            dir = self.root / sequence / "Databases" / "Images"
            views = [p.stem for p in sorted(dir.glob("*.lmdb"))]
            views = [view for view in views if view in self.views]
            for view in tqdm(views, leave=False):
                dataset = PanopticPoseDataset(
                    path=self.root / sequence,
                    view=view,
                )
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset


class PanopticImageDataModule(DataModule):
    def __init__(
        self,
        root: Union[str, Path],
        sequences: Iterable[str] = KNOWN_SEQUENCES,
        train_intervals: Iterable = TRAIN_KNOWN_INTERVALS,
        val_intervals: Iterable = VAL_KNOWN_INTERVALS,
        test_intervals: Iterable = TEST_KNOWN_INTERVALS,
        views: Iterable[str] = VIEWS,
        cr_size: Union[int, Tuple[int, int]] = None,
        re_size: Union[int, Tuple[int, int]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super(PanopticImageDataModule, self).__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.sequences = sequences
        self.train_intervals = train_intervals
        self.val_intervals = val_intervals
        self.test_intervals = test_intervals
        self.views = views

        self.transform = PanopticImageTransform(
            cr_size=cr_size,
            re_size=re_size,
        )

    def _train_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.train_intervals):
            dir = self.root / sequence / "Databases" / "Images"
            views = [p.stem for p in sorted(dir.glob("*.lmdb"))]
            views = [view for view in views if view in self.views]
            for view in tqdm(views, leave=False):
                dataset = PanopticImageDataset(
                    path=self.root / sequence,
                    view=view,
                    transform=self.transform,
                )
                dataset.keys = filter(dataset.keys, intervals)
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset

    def _val_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.val_intervals):
            dir = self.root / sequence / "Databases" / "Images"
            views = [p.stem for p in sorted(dir.glob("*.lmdb"))]
            views = [view for view in views if view in self.views]
            for view in tqdm(views, leave=False):
                dataset = PanopticImageDataset(
                    path=self.root / sequence,
                    view=view,
                    transform=self.transform,
                )
                dataset.keys = filter(dataset.keys, intervals, mod=30)
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset

    def _test_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.test_intervals):
            dir = self.root / sequence / "Databases" / "Images"
            views = [p.stem for p in sorted(dir.glob("*.lmdb"))]
            views = [view for view in views if view in self.views]
            for view in tqdm(views, leave=False):
                dataset = PanopticImageDataset(
                    path=self.root / sequence,
                    view=view,
                    transform=self.transform,
                )
                dataset.keys = filter(dataset.keys, intervals, mod=30)
                datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        return dataset


class PanopticImagePairDataModule(DataModule):
    def __init__(
        self,
        root: Union[str, Path],
        sequences: Iterable[str] = KNOWN_SEQUENCES,
        train_intervals: Iterable = TRAIN_KNOWN_INTERVALS,
        val_intervals: Iterable = VAL_KNOWN_INTERVALS,
        test_intervals: Iterable = TEST_KNOWN_INTERVALS,
        in_views: Iterable[str] = VIEWS,
        out_views: Iterable[str] = VIEWS,
        cr_size_A: Union[int, Tuple[int, int]] = None,
        cr_size_B: Union[int, Tuple[int, int]] = None,
        re_size_A: Union[int, Tuple[int, int]] = None,
        re_size_B: Union[int, Tuple[int, int]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super(PanopticImagePairDataModule, self).__init__(
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

        self.transform = PanopticImagePairTransform(
            cr_size_A=cr_size_A,
            cr_size_B=cr_size_B,
            re_size_A=re_size_A,
            re_size_B=re_size_B,
        )

    def _train_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.train_intervals):
            dataset = PanopticImagePairDataset(
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
            dataset = PanopticImagePairDataset(
                path=self.root / sequence,
                in_views=self.in_views,
                out_views=self.out_views,
                transform=self.transform,
            )
            dataset.keys = filter(dataset.keys, intervals, mod=30)
            datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets

    def _test_dataset(self):
        datasets = []
        for sequence, intervals in zip(tqdm(self.sequences), self.test_intervals):
            dataset = PanopticImagePairDataset(
                path=self.root / sequence,
                in_views=self.in_views,
                out_views=self.out_views,
                transform=self.transform,
            )
            dataset.keys = filter(dataset.keys, intervals, mod=30)
            datasets.append(dataset)
        datasets = ConcatDataset(datasets)
        return datasets
