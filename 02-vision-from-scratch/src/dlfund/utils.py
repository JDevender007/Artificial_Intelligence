"""Seeding, device and logging helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    With ``deterministic=True`` cuDNN autotuning is disabled and deterministic kernels
    are requested, which makes GPU runs bitwise repeatable at some speed cost. Without
    it, runs are repeatable up to tiny floating-point differences.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - DataLoader signature
    """DataLoader ``worker_init_fn``: derive NumPy/random seeds from the torch seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "dlfund", log_file: Path | None = None) -> logging.Logger:
    """Return a logger that writes to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def count_params(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
