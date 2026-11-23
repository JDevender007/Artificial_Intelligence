"""Download MNIST and CIFAR-10 via torchvision and print verifiable stats.

    python scripts/download_data.py --data-dir data
    python scripts/download_data.py --datasets mnist      # MNIST only (~11 MB)
"""

from __future__ import annotations

import argparse
from collections import Counter

from torchvision import datasets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"], choices=["mnist", "cifar10"])
    args = ap.parse_args()
    available = {"mnist": ("MNIST", datasets.MNIST), "cifar10": ("CIFAR-10", datasets.CIFAR10)}
    for name, cls in (available[k] for k in args.datasets):
        for train in (True, False):
            ds = cls(args.data_dir, train=train, download=True)
            counts = Counter(int(t) for t in ds.targets)
            shape = tuple(ds.data.shape)
            print(f"{name:9s} {'train' if train else 'test ':5s} n={len(ds):6d} shape={shape} "
                  f"per-class min/max={min(counts.values())}/{max(counts.values())}")


if __name__ == "__main__":
    main()
