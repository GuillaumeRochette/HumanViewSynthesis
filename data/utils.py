import json
from itertools import groupby
from pathlib import Path
from random import randrange

import torch


def split(n: int, p: float = 0.1):
    assert 0.0 < p < 0.5
    m = int(p * n)
    i = randrange(0, n - m)
    j = i
    while i - m < j < i + m:
        j = randrange(0, n - m)

    if i < j:
        train_split = [(0, i), (i + m, j), (j + m, n)]
    else:
        train_split = [(0, j), (j + m, i), (i + m, n)]

    val_split = [(i, i + m)]
    test_split = [(j, j + m)]

    return train_split, val_split, test_split


def load_camera(path: Path, view: str):
    with path.open() as file:
        cameras = json.load(file)
    camera = {
        "R": torch.tensor(cameras[view]["R"]),
        "t": torch.tensor(cameras[view]["t"]),
        "K": torch.tensor(cameras[view]["K"]),
        "dist_coef": torch.tensor(cameras[view]["dist_coef"]),
        "resolution": torch.tensor(cameras[view]["resolution"]),
    }
    return camera


def load_cameras(path: Path):
    with path.open() as file:
        cameras = json.load(file)
    cameras = {
        view: {
            "R": torch.tensor(cameras[view]["R"]),
            "t": torch.tensor(cameras[view]["t"]),
            "K": torch.tensor(cameras[view]["K"]),
            "dist_coef": torch.tensor(cameras[view]["dist_coef"]),
            "resolution": torch.tensor(cameras[view]["resolution"]),
        }
        for view in cameras
    }
    return cameras


def filter(keys, intervals, mod=1):
    keys = [k for s, e in intervals for k in keys if s <= k <= e]
    keys = [k for k in keys if k % mod == 0]

    return keys
