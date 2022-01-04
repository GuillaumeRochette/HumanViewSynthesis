from torch import Tensor

from geometry import matrix


def world_to_camera(xyz: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """
    Transform 3D coordinates from world view into camera view.

    :param xyz: 3D coordinates of shape [*, 3, 1].
    :param R: Rotation matrix of shape [*, 3, 3].
    :param t: Translation vector of shape [*, 3, 1].
    :return: 3D coordinates of shape [*, 3, 1].
    """
    assert xyz.ndim == R.ndim == t.ndim
    assert xyz.shape[-2:] == (3, 1)
    assert R.shape[-2:] == (3, 3)
    assert t.shape[-2:] == (3, 1)

    xyz = matrix.affine(x=xyz, A=R, b=t)

    return xyz


def camera_to_world(xyz: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """
    Transform 3D coordinates from camera view into world view.

    :param xyz: 3D coordinates of shape [*, 3, 1].
    :param R: Rotation matrix of shape [*, 3, 3].
    :param t: Translation vector of shape [*, 3, 1].
    :return: 3D coordinates of shape [*, 3, 1].
    """
    assert xyz.ndim == R.ndim == t.ndim
    assert xyz.shape[-2:] == (3, 1)
    assert R.shape[-2:] == (3, 3)
    assert t.shape[-2:] == (3, 1)

    A = R.transpose(-1, -2)
    b = -A @ t
    xyz = matrix.affine(x=xyz, A=A, b=b)

    return xyz


def camera_to_camera(
    xyz: Tensor,
    R1: Tensor,
    t1: Tensor,
    R2: Tensor,
    t2: Tensor,
) -> Tensor:
    """
    Transform 3D coordinates from the first view into second view.

    :param xyz: 3D coordinates of shape [*, 3, 1].
    :param R1: Rotation matrix of shape [*, 3, 3].
    :param t1: Translation vector of shape [*, 3, 1].
    :param R2: Rotation matrix of shape [*, 3, 3].
    :param t2: Translation vector of shape [*, 3, 1].
    :return: 3D coordinates of shape [*, 3, 1].
    """
    assert xyz.ndim == R1.ndim == t1.ndim == R2.ndim == t2.ndim
    assert xyz.shape[-2:] == (3, 1)
    assert R1.shape[-2:] == R2.shape[-2:] == (3, 3)
    assert t1.shape[-2:] == t2.shape[-2:] == (3, 1)

    A = R2 @ R1.transpose(-1, -2)
    b = -A @ t1 + t2
    xyz = matrix.affine(x=xyz, A=A, b=b)

    return xyz
