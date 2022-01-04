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
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--in_view", type=str, required=True)
    parser.add_argument("--out_view", type=str, required=True)
    parser.add_argument("--interval", type=int, nargs=2, default=[None, None])
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
        from data.Panoptic.dataset import PanopticImagePairDataset
        from data.Panoptic.transform import PanopticImagePairTransform

        dataset = PanopticImagePairDataset(
            path=args.root / args.sequence,
            in_views=[args.in_view],
            out_views=[args.out_view],
            transform=PanopticImagePairTransform(
                cr_size_A=1080,
                cr_size_B=1080,
                re_size_A=litmodel.hparams.input_resolution,
                re_size_B=litmodel.hparams.target_resolution,
            ),
        )
    elif "Human3.6M" in str(args.root):
        from data.Human36M.dataset import Human36MImagePairDataset
        from data.Human36M.transform import Human36MImagePairTransform

        dataset = Human36MImagePairDataset(
            path=args.root / args.sequence,
            in_views=[args.in_view],
            out_views=[args.out_view],
            transform=Human36MImagePairTransform(
                cr_margin_A=0.25,
                cr_margin_B=0.25,
                re_size_A=litmodel.hparams.input_resolution,
                re_size_B=litmodel.hparams.target_resolution,
            ),
        )
    else:
        raise ValueError(f"Unknown root: {args.root}")

    s, e = args.interval
    if s is None:
        s = min(dataset.keys)
    if e is None:
        e = max(dataset.keys)
    dataset.keys = [k for k in dataset.keys if s <= k <= e]
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(len(dataloader))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / f"TMP_{time()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    for input in tqdm(dataloader):
        input = RecursiveMunch.fromDict(input)

        input = move(input, gpu)

        output = litmodel.model.infer_real(input)

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
