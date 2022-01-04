import argparse

import torch

from geometry import vector, rotation
from rendering.integral_raycasting import IntegralRayCasting
from utils.plot import get_palette, show


def define_camera(h: int, w: int):
    f_x, f_y = 1600.0, 1600.0
    c_x, c_y = 960.0, 540.0

    w_r, h_r = w / 1920, h / 1080

    f_x, f_y = w_r * f_x, h_r * f_y
    c_x, c_y = w_r * c_x, h_r * c_y

    K = torch.tensor(
        [
            [f_x, 0.0, c_x],
            [0.0, f_y, c_y],
            [0.0, 0.0, 1.0],
        ]
    )
    k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0
    dist_coef = torch.tensor([k1, k2, p1, p2, k3])

    return K, dist_coef


def one_sphere(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 4.0,
    radius: float = 0.5,
):
    mu = torch.tensor([x, y, z]).reshape(1, 3, 1)

    rho = torch.eye(3).reshape(1, 3, 3)

    lambd = torch.tensor([radius, radius, radius]).square().reshape(1, 3, 1)

    appearance = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3)

    background_appearance = torch.zeros(1, 3)

    return mu, rho, lambd, appearance, background_appearance


def one_ellipsoid(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 4.0,
    major_axis: float = 0.5,
    minor_axis: float = 0.15,
):
    mu = torch.tensor([x, y, z]).reshape(1, 3, 1)

    d = torch.randn(1, 3, 1)
    e = torch.eye(3).split(1, dim=-1)[0].expand(1, 3, 1)
    rho = rotation.vector_rotation(x=d, y=e)

    lambd = torch.tensor([major_axis, minor_axis, minor_axis]).square().reshape(1, 3, 1)

    appearance = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3)

    background_appearance = torch.zeros(1, 3)

    return mu, rho, lambd, appearance, background_appearance


def two_spheres(
    x_1: float = 0.0,
    y_1: float = 0.0,
    z_1: float = 3.0,
    x_2: float = 0.0,
    y_2: float = 0.125,
    z_2: float = 4.0,
    radius_1: float = 0.1,
    radius_2: float = 0.5,
):
    mu = torch.tensor([[x_1, y_1, z_1], [x_2, y_2, z_2]]).reshape(2, 3, 1)

    rho = torch.eye(3).reshape(1, 3, 3).expand(2, 3, 3)

    lambd = (
        torch.tensor([[radius_1, radius_1, radius_1], [radius_2, radius_2, radius_2]])
        .square()
        .reshape(2, 3, 1)
    )

    appearance = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).reshape(2, 3)

    background_appearance = torch.zeros(1, 3)

    return mu, rho, lambd, appearance, background_appearance


def many_spheres(
    n: int,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 4.0,
    radius: float = 0.1,
):
    mu = torch.tensor([x, y, z]).reshape(1, 3, 1) + vector.normalize(
        x=torch.randn(n, 3, 1)
    )

    rho = torch.eye(3).reshape(1, 3, 3).expand(n, 3, 3)

    lambd = (
        torch.tensor([radius, radius, radius]).square().reshape(1, 3, 1).expand(n, 3, 1)
    )

    appearance = get_palette(n)

    background_appearance = torch.zeros(1, 3)

    return mu, rho, lambd, appearance, background_appearance


def many_ellipsoids(
    n: int,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 4.0,
    major_axis: float = 0.2,
    minor_axis: float = 0.05,
):
    mu = torch.tensor([x, y, z]).reshape(1, 3, 1) + vector.normalize(
        x=torch.randn(n, 3, 1)
    )

    d = torch.randn(n, 3, 1)
    e = torch.eye(3).split(1, dim=-1)[0].reshape(1, 3, 1).expand(n, 3, 1)
    rho = rotation.vector_rotation(x=d, y=e)

    lambd = (
        torch.tensor([major_axis, minor_axis, minor_axis])
        .square()
        .reshape(1, 3, 1)
        .expand(n, 3, 1)
    )

    appearance = get_palette(n)

    background_appearance = torch.zeros(1, 3)

    return mu, rho, lambd, appearance, background_appearance


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--one_sphere", action="store_true")
    group.add_argument("--one_ellipsoid", action="store_true")
    group.add_argument("--two_spheres", action="store_true")
    group.add_argument("--many_spheres", type=int)
    group.add_argument("--many_ellipsoids", type=int)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    K, dist_coef = define_camera(h=args.height, w=args.width)
    if args.one_sphere:
        mu, rho, lambd, appearance, background_appearance = one_sphere()
    elif args.one_ellipsoid:
        mu, rho, lambd, appearance, background_appearance = one_ellipsoid()
    elif args.two_spheres:
        mu, rho, lambd, appearance, background_appearance = two_spheres()
    elif args.many_spheres:
        mu, rho, lambd, appearance, background_appearance = many_spheres(n=args.many_spheres)
    elif args.many_ellipsoids:
        mu, rho, lambd, appearance, background_appearance = many_ellipsoids(n=args.many_ellipsoids)
    else:
        raise ValueError()

    renderer = IntegralRayCasting(resolution=(args.height, args.width))

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

    print("image", image.shape)

    show(image)


if __name__ == "__main__":
    main()
