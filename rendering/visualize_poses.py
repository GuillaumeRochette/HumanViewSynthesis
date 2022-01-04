import argparse
from munch import RecursiveMunch
from pathlib import Path

import torch

from geometry.primitives import Pose3DToLimbs
from rendering.integral_raycasting import IntegralRayCasting
from utils.plot import get_palette, show


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--view", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if "Panoptic" in str(args.root):
        from data.Panoptic.dataset import PanopticImageDataset
        from data.Panoptic.transform import PanopticImageTransform
        from data.Panoptic.skeleton import EDGES
        from data.Panoptic.statistics import WIDTHS

        dataset = PanopticImageDataset(
            path=args.root / args.sequence,
            view=args.view,
            transform=PanopticImageTransform(
                cr_size=1080,
                re_size=(args.height, args.width),
            ),
        )
    elif "Human3.6M" in str(args.root):
        from data.Human36M.dataset import Human36MImageDataset
        from data.Human36M.transform import Human36MImageTransform
        from data.Human36M.skeleton import EDGES
        from data.Human36M.statistics import WIDTHS

        dataset = Human36MImageDataset(
            path=args.root / args.sequence,
            view=args.view,
            transform=Human36MImageTransform(
                cr_margin=0.25,
                re_size=(args.height, args.width),
            ),
        )
    else:
        raise ValueError(f"Unknown root: {args.root}")

    p2p = Pose3DToLimbs(edges=EDGES, widths=WIDTHS)
    renderer = IntegralRayCasting(resolution=(args.height, args.width))

    dataset.keys = dataset.keys[args.frame: args.frame + 1]
    input = RecursiveMunch.fromDict(dataset[0])

    input.pose_3d.p = input.pose_3d.root.p + input.pose_3d.relative.p

    mu, rho, lambd = p2p(input.pose_3d.p)
    appearance = get_palette(len(EDGES))
    background_appearance = torch.zeros(1, 3)
    K = input.K
    dist_coef = input.dist_coef

    print("mu", mu.shape)
    print("rho", rho.shape)
    print("lambd", lambd.shape)
    print("appearance", appearance.shape)
    print("background_appearance", background_appearance.shape)
    print("K", K.shape)
    print("dist_coef", dist_coef.shape)

    image = renderer.to(device=device)(
        mu=mu.to(device=device),
        rho=rho.to(device=device),
        lambd=lambd.to(device=device),
        appearance=appearance.to(device=device),
        background_appearance=background_appearance.to(device=device),
        K=K.to(device=device),
        dist_coef=dist_coef.to(device=device),
    )
    print(image.shape)

    show(image)


if __name__ == "__main__":
    main()
