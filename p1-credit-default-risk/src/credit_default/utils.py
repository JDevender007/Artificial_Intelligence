"""Seeding and logging helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and the hash seed so a run is reproducible.

    Model libraries (XGBoost, LightGBM, Optuna) receive the seed explicitly
    through their ``random_state`` / sampler arguments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "credit_default", log_file: Path | None = None) -> logging.Logger:
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
