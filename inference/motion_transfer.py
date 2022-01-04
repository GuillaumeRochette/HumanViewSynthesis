import argparse
from munch import RecursiveMunch
import shutil
import subprocess
from tqdm import tqdm
from time import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from lightning.novel_view_synthesis import LitNovelViewSynthesis
from utils.inference import move
from utils.plot import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence_motion", type=str, required=True)
    parser.add_argument("--in_view_motion", type=str, required=True)
    parser.add_argument("--out_view_motion", type=str, required=True)
    parser.add_argument("--interval_motion", type=int, nargs=2, default=[None, None])
    parser.add_argument("--sequence_appearance", type=str, required=True)
    parser.add_argument("--in_view_appearance", type=str, required=True)
    parser.add_argument("--interval_appearance", type=int, nargs=2, default=[None, None])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30000 / 1001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cpu = torch.device("cpu")
    if torch.cuda.is_available():
        gpu = torch.device("cuda")
    else:
        gpu = torch.device("cpu")

    litmodel = LitNovelViewSynthesis.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        strict=False,
    )
    litmodel.freeze()
    litmodel = litmodel.to(device=gpu)

    if "Panoptic" in str(args.root):
        from data.Panoptic.dataset import PanopticImageDataset, PanopticImagePairDataset
        from data.Panoptic.transform import (
            PanopticImageTransform,
            PanopticImagePairTransform,
        )

        dataset_motion = PanopticImagePairDataset(
            path=args.root / args.sequence_motion,
            in_views=[args.in_view_motion],
            out_views=[args.out_view_motion],
            transform=PanopticImagePairTransform(
                cr_size_A=1080,
                cr_size_B=1080,
                re_size_A=litmodel.hparams.input_resolution,
                re_size_B=litmodel.hparams.target_resolution,
            ),
        )
        dataset_appearance = PanopticImageDataset(
            path=args.root / args.sequence_appearance,
            view=args.in_view_appearance,
            transform=PanopticImageTransform(
                cr_size=1080,
                re_size=litmodel.hparams.input_resolution,
            ),
        )
    elif "Human3.6M" in str(args.root):
        from data.Human36M.dataset import Human36MImageDataset, Human36MImagePairDataset
        from data.Human36M.transform import (
            Human36MImageTransform,
            Human36MImagePairTransform,
        )

        dataset_motion = Human36MImagePairDataset(
            path=args.root / args.sequence_motion,
            in_views=[args.in_view_motion],
            out_views=[args.out_view_motion],
            transform=Human36MImagePairTransform(
                cr_margin_A=0.25,
                cr_margin_B=0.25,
                re_size_A=litmodel.hparams.input_resolution,
                re_size_B=litmodel.hparams.target_resolution,
            ),
        )
        dataset_appearance = Human36MImageDataset(
            path=args.root / args.sequence_appearance,
            view=args.in_view_appearance,
            transform=Human36MImageTransform(
                cr_margin=0.25,
                re_size=litmodel.hparams.input_resolution,
            ),
        )
    else:
        raise ValueError(f"Unknown root: {args.root}")

    s, e = args.interval_motion
    if s is None:
        s = min(dataset_motion.keys)
    if e is None:
        e = max(dataset_motion.keys)
    dataset_motion.keys = [k for k in dataset_motion.keys if s <= k <= e]
    dataloader_motion = DataLoader(
        dataset_motion,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    s, e = args.interval_appearance
    if s is None:
        s = min(dataset_appearance.keys)
    if e is None:
        e = max(dataset_appearance.keys)
    dataset_appearance.keys = [k for k in dataset_appearance.keys if s <= k <= e]
    dataloader_appearance = DataLoader(
        dataset_appearance,
        batch_size=1,
        num_workers=0
    )

    print(len(dataloader_motion))
    print(len(dataloader_appearance))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / f"TMP_{time()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    for input in dataloader_appearance:
        input = RecursiveMunch.fromDict(input)
        image_C = input.image
        pad_C = input.stabilized_padding

    for input in tqdm(dataloader_motion):
        input = RecursiveMunch.fromDict(input)

        input.C.image = image_C.clone()
        input.C.stabilized_padding = pad_C.clone()

        input = move(input, gpu)

        output = litmodel.model.motion_transfer(input)

        input = move(input, cpu)
        output = move(output, cpu)

        for i, key in enumerate(input.key):
            name = f"{int(key):012d}.png"

            save(
                image=F.pad(
                    input.A.image[i],
                    pad=input.A.stabilized_padding[i].tolist(),
                    value=1.0,
                ),
                path=tmp_dir / "Input_A" / name,
            )
            save(
                image=F.pad(
                    input.C.image[i],
                    pad=input.C.stabilized_padding[i].tolist(),
                    value=1.0,
                ),
                path=tmp_dir / "Input_C" / name,
            )
            save(
                image=F.pad(
                    output.B.image[i].clamp(min=0.0, max=1.0),
                    pad=input.B.stabilized_padding[i].tolist(),
                    value=1.0,
                ),
                path=tmp_dir / "Output_B" / name,
            )
            save(
                image=F.pad(
                    input.B.image[i],
                    pad=input.B.stabilized_padding[i].tolist(),
                    value=1.0,
                ),
                path=tmp_dir / "Input_B" / name,
            )

    input = None
    output = None
    litmodel = None

    for image_dir in sorted(tmp_dir.glob("*")):
        output_video = args.output_dir / f"{image_dir.stem}.mp4"
        cmd = f"/usr/bin/ffmpeg -f image2 -pattern_type glob -framerate {args.fps} -i {image_dir}/*.png -pix_fmt yuv420p -crf 0 -y {output_video}"
        print(cmd)
        process = subprocess.run(cmd.split(), check=True)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
