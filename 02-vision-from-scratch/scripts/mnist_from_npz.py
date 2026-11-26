"""Offline fallback: write torchvision's raw MNIST files from a local Keras ``mnist.npz``.

Use only when the torchvision download is too slow. Keras ships the same 60k/10k MNIST
split; the per-class counts printed by ``download_data.py`` must match the official ones.

    python scripts/mnist_from_npz.py --npz ~/.keras/datasets/mnist.npz --data-dir data
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def _write_idx(path: Path, arr: np.ndarray) -> None:
    magic = 0x0803 if arr.ndim == 3 else 0x0801
    with open(path, "wb") as f:
        f.write(struct.pack(">I", magic))
        for d in arr.shape:
            f.write(struct.pack(">I", d))
        f.write(arr.astype(np.uint8).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    z = np.load(args.npz)
    raw = Path(args.data_dir) / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _write_idx(raw / "train-images-idx3-ubyte", z["x_train"])
    _write_idx(raw / "train-labels-idx1-ubyte", z["y_train"])
    _write_idx(raw / "t10k-images-idx3-ubyte", z["x_test"])
    _write_idx(raw / "t10k-labels-idx1-ubyte", z["y_test"])
    print(f"Wrote {raw}: train {z['x_train'].shape}, test {z['x_test'].shape}")


if __name__ == "__main__":
    main()
