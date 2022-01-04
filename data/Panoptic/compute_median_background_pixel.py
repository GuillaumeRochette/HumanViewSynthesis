import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np

from data.Panoptic.datamodule import PanopticImageDataModule


class SpecialTransform(object):
    def __call__(self, input: dict) -> dict:
        input["image"] = np.array(input["image"], dtype=np.uint8)

        input["mask"] = np.array(input["mask"], dtype=np.bool)[..., None]

        input = {
            "image": input["image"],
            "mask": input["mask"],
        }

        return input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    dm = PanopticImageDataModule(
        root=args.root,
        batch_size=1,
        num_workers=8,
    )

    dm.transform = SpecialTransform()

    dm.setup()

    counts = np.zeros((3, 256), dtype=np.uint64)

    for j, input in enumerate(tqdm(dm.train_dataloader())):
        image = input["image"].numpy().reshape((-1, 3))
        mask = input["mask"].numpy().reshape((-1))

        for i, channel in enumerate(np.split(image, 3, axis=-1)):
            x = channel[~mask]
            v, c = np.unique(x, return_counts=True)
            v = v.astype(np.uint64)
            c = c.astype(np.uint64)
            counts[i, v] += c

        if j == 10000:
            break

    half = (np.sum(counts, axis=-1, keepdims=True) + 1) // 2
    cumsum = np.cumsum(counts, axis=-1)
    condition = cumsum < half
    pixel = np.sum(condition, axis=-1)
    print(pixel)


if __name__ == "__main__":
    main()
