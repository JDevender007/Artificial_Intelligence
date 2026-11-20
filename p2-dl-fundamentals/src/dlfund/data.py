"""Datasets, transforms and seeded DataLoaders.

The official train split is divided into train/val with a FIXED ``split_seed`` so every
training seed and every model sees identical data; the official test split is used
once, after training, for the reported numbers. Early stopping and LR choices only
ever look at val.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from dlfund.config import DataConfig
from dlfund.utils import seed_worker

STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "fake": ((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLASSES: dict[str, list[str]] = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"],
    "fake": [f"c{i}" for i in range(10)],
}
NATIVE = {"mnist": (1, 28), "cifar10": (3, 32), "fake": (3, 32)}  # (channels, size)


@dataclass
class DataInfo:
    """What a model needs to know about its input."""

    in_channels: int
    img_size: int
    n_classes: int
    classes: list[str]


def build_transforms(dataset: str, model: str, augment: bool, train: bool,
                     resnet_img_size: int = 128) -> transforms.Compose:
    """Transforms for one (dataset, model) pair.

    The pretrained ResNet gets 3-channel, upsampled, ImageNet-normalised input (that is
    the distribution its weights were trained on); the from-scratch models get native
    resolution with dataset statistics.
    """
    ops: list = []
    if train and augment:
        if dataset == "mnist":
            ops.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
        else:  # cifar10 / fake: standard crop + flip
            ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if model == "resnet18":
        if dataset == "mnist":
            ops.append(transforms.Grayscale(num_output_channels=3))
        ops.append(transforms.Resize(resnet_img_size, antialias=True))
        mean, std = IMAGENET
    else:
        mean, std = STATS[dataset]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def _raw(dataset: str, root: Path, train: bool, transform, download: bool = True) -> Dataset:
    if dataset == "mnist":
        return datasets.MNIST(root, train=train, transform=transform, download=download)
    if dataset == "cifar10":
        return datasets.CIFAR10(root, train=train, transform=transform, download=download)
    # Synthetic images for tests/CI: no download, labels are random.
    return datasets.FakeData(size=600 if train else 200, image_size=(3, 32, 32), num_classes=10,
                             transform=transform, random_offset=0 if train else 10_000)


def split_indices(n: int, val_size: int, split_seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val index split of ``range(n)``."""
    if not 0 < val_size < n:
        raise ValueError(f"val_size must be in (0, {n}), got {val_size}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed)).tolist()
    return perm[val_size:], perm[:val_size]


def get_datasets(cfg: DataConfig, model: str) -> tuple[Dataset, Dataset, Dataset, DataInfo]:
    """Return ``(train, val, test, info)`` for ``cfg.dataset`` as seen by ``model``."""
    root = Path(cfg.data_dir)
    tf_train = build_transforms(cfg.dataset, model, cfg.augment, True, cfg.resnet_img_size)
    tf_eval = build_transforms(cfg.dataset, model, cfg.augment, False, cfg.resnet_img_size)
    full_train = _raw(cfg.dataset, root, True, tf_train)
    full_eval = _raw(cfg.dataset, root, True, tf_eval)  # same images, no augmentation
    test = _raw(cfg.dataset, root, False, tf_eval)

    val_size = cfg.val_size if cfg.dataset != "fake" else 100
    tr_idx, va_idx = split_indices(len(full_train), val_size, cfg.split_seed)
    if cfg.limit_train is not None:
        tr_idx = tr_idx[: cfg.limit_train]

    ch, size = NATIVE[cfg.dataset]
    if model == "resnet18":
        ch, size = 3, cfg.resnet_img_size
    info = DataInfo(ch, size, len(CLASSES[cfg.dataset]), CLASSES[cfg.dataset])
    return Subset(full_train, tr_idx), Subset(full_eval, va_idx), test, info


def make_loaders(train: Dataset, val: Dataset, test: Dataset, batch_size: int, num_workers: int,
                 seed: int, pin_memory: bool) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Seeded loaders: the shuffle order and augmentation RNG depend only on ``seed``.

    Only the training loader (random augmentation, the bottleneck) uses worker processes
    on Windows. There, every worker is *spawned* and reloads torch's CUDA DLLs (several
    hundred MB of committed memory each); six persistent workers across three loaders
    exhausted the paging file on a 16 GB laptop (``WinError 1455``).
    """
    eval_workers = eval_loader_workers(num_workers)
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
                    **_loader_kwargs(num_workers, pin_memory))
    va = DataLoader(val, batch_size=batch_size * 2, shuffle=False,
                    **_loader_kwargs(eval_workers, pin_memory))
    te = DataLoader(test, batch_size=batch_size * 2, shuffle=False,
                    **_loader_kwargs(eval_workers, pin_memory))
    return tr, va, te


def eval_loader_workers(num_workers: int, platform: str = os.name) -> int:
    """Workers for the val/test loaders: none on Windows (spawn cost), else ``num_workers``."""
    return 0 if platform == "nt" else num_workers


def _loader_kwargs(num_workers: int, pin_memory: bool) -> dict:
    return {"num_workers": num_workers, "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
            "worker_init_fn": seed_worker if num_workers > 0 else None}
