import subprocess
import sys

from torch import Tensor


def git_hash():
    cmd = "git rev-parse HEAD"
    output = subprocess.check_output(args=cmd.split(), universal_newlines=True)
    hash = output.strip()
    return hash


def check_finiteness(d: dict) -> bool:
    finiteness = True
    for k, v in d.items():
        if isinstance(v, dict):
            finiteness &= check_finiteness(d=v)
        elif isinstance(v, Tensor):
            finiteness &= v.isfinite().all().tolist()
    return finiteness


def summarize_dict(d: dict) -> dict:
    d_ = {}
    for k, v in d.items():
        if isinstance(v, dict):
            d_[k] = summarize_dict(d=v)
        elif isinstance(v, Tensor):
            d_[k] = {
                "shape": tuple(v.shape),
                "dtype": v.dtype,
                "requires_grad": v.requires_grad,
                "[min, max]": [v.min().tolist(), v.max().tolist()],
                "[mean, std]": [v.float().mean().tolist(), v.float().std().tolist()],
            }
        else:
            d_[k] = v
    return d_


def log_finiteness(input, output, losses):
    if not check_finiteness(losses):
        print("input =", summarize_dict(input), file=sys.stderr)
        print("output =", summarize_dict(output), file=sys.stderr)
        print("losses =", summarize_dict(losses), file=sys.stderr)


def flatten_dict(d: dict, k_="", sep="_") -> dict:
    d_ = {}
    for k, v in d.items():
        if k_:
            k__ = k_ + sep + k
        else:
            k__ = k
        if isinstance(v, dict):
            d_.update(flatten_dict(d=v, k_=k__, sep=sep))
        else:
            d_[k__] = v
    return d_
