from typing import Union, Iterable
from pathlib import Path

import torch

from data.dataset import PoseDataset, ImageDataset, ImagePairDataset
from data.database import ImageDatabase, LabelDatabase, MaskDatabase
from data.utils import load_camera, load_cameras
from data.Human36M.skeleton import JOINTS

from geometry.extrinsic import world_to_camera


class Human36MPoseDataset(PoseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        view: str,
    ):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.view = view

        p = path / "Databases" / "Poses" / "2D" / "CPN" / f"{view}.lmdb"
        poses_2d = LabelDatabase(path=p)
        p = path / "Databases" / "Poses" / "3D" / "Groundtruth.lmdb"
        poses_3d = LabelDatabase(path=p)
        p = path / "cameras.json"
        self.camera = load_camera(path=p, view=view)

        self.keys = sorted(set(poses_2d.keys).intersection(poses_3d.keys))
        self.indexes = {k: i for i, k in enumerate(self.keys)}

        p = torch.stack(poses_2d[self.keys])
        c = torch.ones(len(p), len(JOINTS), 1, 1, dtype=torch.bool)
        self.poses_2d = {
            "p": p,
            "c": c,
        }

        p = torch.stack(poses_3d[self.keys])
        c = torch.ones(len(p), len(JOINTS), 1, 1, dtype=torch.bool)
        p = world_to_camera(
            xyz=p,
            R=self.camera["R"][None, None, :, :],
            t=self.camera["t"][None, None, :, :],
        )
        self.poses_3d = {
            "root": {
                "p": p[:, JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                "c": c[:, JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
            },
            "relative": {
                "p": p - p[:, JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
                "c": c & c[:, JOINTS["HipCenter"] : JOINTS["HipCenter"] + 1, :, :],
            },
        }


class Human36MImageDataset(ImageDataset):
    def __init__(
        self,
        path: Union[str, Path],
        view: str,
        transform=None,
    ):
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.view = view

        p = path / "Databases" / "Images" / f"{view}.lmdb"
        self.images = ImageDatabase(path=p)
        p = path / "Databases" / "Masks" / "SOLO" / f"{view}.lmdb"
        self.masks = MaskDatabase(path=p)
        p = path / "Databases" / "Poses" / "2D" / "CPN" / f"{view}.lmdb"
        self.poses_2d = LabelDatabase(path=p)
        p = path / "Databases" / "Poses" / "3D" / "Groundtruth.lmdb"
        self.poses_3d = LabelDatabase(path=p)
        p = path / "cameras.json"
        self.camera = load_camera(path=p, view=view)

        self.keys = self._keys()

        self.transform = transform

    def _keys(self):
        """
        Computes the intersection between images and poses_3d databases to retrieve valid keys.

        :return: List of valid keys.
        """
        keys = set(self.images.keys).intersection(
            self.masks.keys,
            self.poses_2d.keys,
            self.poses_3d.keys,
        )
        return sorted(keys)


class Human36MImagePairDataset(ImagePairDataset):
    def __init__(
        self,
        path: Union[str, Path],
        in_views: Iterable[str] = None,
        out_views: Iterable[str] = None,
        transform=None,
    ):
        if not isinstance(path, Path):
            path = Path(path)

        images_dir = path / "Databases" / "Images"
        images = {p.stem: p for p in images_dir.glob("*.lmdb")}
        masks_dir = path / "Databases" / "Masks" / "SOLO"
        masks = {p.stem: p for p in masks_dir.glob("*.lmdb")}
        poses_2d_dir = path / "Databases" / "Poses" / "2D" / "CPN"
        poses_2d = {p.stem: p for p in poses_2d_dir.glob("*.lmdb")}
        poses_3d_dir = path / "Databases" / "Poses" / "3D"
        poses_3d = poses_3d_dir / "Groundtruth.lmdb"

        if in_views is None:
            in_views = {view for view in images}
        else:
            in_views = {view for view in images if view in in_views}

        if out_views is None:
            out_views = {view for view in images}
        else:
            out_views = {view for view in images if view in out_views}

        views = in_views.union(out_views)
        assert views <= images.keys() == masks.keys() == poses_2d.keys()

        self.path = path
        self.in_views = sorted(in_views)
        self.out_views = sorted(out_views)
        self.views = sorted(views)

        self.images = {k: ImageDatabase(path=v) for k, v in images.items()}
        self.masks = {k: MaskDatabase(path=v) for k, v in masks.items()}
        self.poses_2d = {k: LabelDatabase(path=v) for k, v in poses_2d.items()}
        self.poses_3d = LabelDatabase(path=poses_3d)

        self.keys = self._keys()
        self.pairs = self._pairs()

        self.cameras = load_cameras(path=path / "cameras.json")

        self.transform = transform

    def _keys(self):
        """
        Computes the intersection between images and poses_3d databases to retrieve valid keys.

        :return: List of valid keys.
        """
        keys = (
            [self.images[view].keys for view in self.views]
            + [self.masks[view].keys for view in self.views]
            + [self.poses_2d[view].keys for view in self.views]
            + [self.poses_3d.keys]
        )
        keys = set(keys[0]).intersection(*(keys[1:]))
        return sorted(keys)

    def _pairs(self):
        """
        Computes pairs of views.

        :return: Pairs of views.
        """
        pairs = [
            [view_A, view_B] for view_A in self.in_views for view_B in self.out_views
        ]
        return pairs
