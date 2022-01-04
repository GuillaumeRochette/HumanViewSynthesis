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

from geometry.rotation import euler_rotation
from math import pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--view", type=str, required=True)
    parser.add_argument("--interval", type=int, nargs=2, default=[None, None])
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30000 / 1001)
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
        from data.Panoptic.dataset import PanopticImageDataset
        from data.Panoptic.transform import PanopticImageTransform

        dataset = PanopticImageDataset(
            path=args.root / args.sequence,
            view=args.view,
            transform=PanopticImageTransform(
                cr_size=1080,
                re_size=litmodel.hparams.input_resolution,
            ),
        )

        w_i, h_i = 1080.0, 1080.0
        w_o = h_o = litmodel.hparams.target_resolution
        f_x, f_y = 1630.0 * w_o / w_i, 1630.0 * h_o / h_i
        c_x, c_y = 540.0 * w_o / w_i, 540.0 * h_o / h_i
        K = [
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0],
        ]
        dist_coef = [-0.220878, 0.189272, 7.79405e-05, 0.000739643, 0.0418043]
        x, y, z = 0.0, 0.0, 2.8
    elif "Human3.6M" in str(args.root):
        from data.Human36M.dataset import Human36MImageDataset
        from data.Human36M.transform import Human36MImageTransform

        dataset = Human36MImageDataset(
            path=args.root / args.sequence,
            view=args.view,
            transform=Human36MImageTransform(
                cr_margin=0.25,
                re_size=litmodel.hparams.input_resolution,
            ),
        )

        w_i, h_i = 1000.0, 1000.0
        w_o = h_o = litmodel.hparams.target_resolution
        f_x, f_y = 1150.0 * w_o / w_i, 1150.0 * h_o / h_i
        c_x, c_y = 500.0 * w_o / w_i, 500.0 * h_o / h_i
        K = [
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0],
        ]
        dist_coef = [
            -0.208338188251856,
            0.255488007488945,
            -0.000759999321030303,
            0.00148438698385668,
            -0.00246049749891915,
        ]
        x, y, z = 0.0, 0.0, 3.0
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
        batch_size=1,
        num_workers=0,
    )

    print(len(dataloader))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.output_dir / f"TMP_{time()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    N = len(dataloader)
    c = 2.0

    anchor = None

    for k, input in enumerate(tqdm(dataloader)):
        input = RecursiveMunch.fromDict(input)

        if anchor is None:
            anchor = input.pose_3d.root.p

        input.pose_3d.root.p = input.pose_3d.root.p - anchor

        R1 = euler_rotation(
            yaw=torch.tensor(c * (k / N) * 2.0 * pi),
            pitch=torch.tensor(0.0),
            roll=torch.tensor(0.0),
        )
        R2 = euler_rotation(
            yaw=torch.tensor(0.0),
            pitch=torch.tensor(0.0),
            roll=torch.tensor(0.0),
        )
        input.R = (R2 @ R1).unsqueeze(dim=0)
        input.t = torch.tensor([[x], [y], [z]]).unsqueeze(dim=0)
        input.K = torch.tensor(K).unsqueeze(dim=0)
        input.dist_coef = torch.tensor(dist_coef).unsqueeze(dim=0)

        input = move(input, gpu)

        output = litmodel.model.infer_virtual(input)

        input = move(input, cpu)
        output = move(output, cpu)

        for i, key in enumerate(input.key):
            name = f"{int(key):012d}.png"

            save(
                image=F.pad(
                    input.image[i],
                    pad=input.stabilized_padding[i].tolist(),
                    value=1.0,
                ),
                path=tmp_dir / "Input_A" / name,
            )
            save(
                image=output.image[i].clamp(min=0.0, max=1.0),
                path=tmp_dir / "Output_B" / name,
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
