"""Inference latency: ms per request at batch size 1 (p50/p95) and ms per text at larger batches."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def measure(predict: Callable[[list[str]], object], texts: list[str], batch_size: int, warmup: int = 5,
            device: str = "cpu") -> dict[str, float]:
    """Time ``predict`` over ``texts`` in chunks of ``batch_size`` after ``warmup`` untimed calls."""
    chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    chunks = [c for c in chunks if len(c) == batch_size] or chunks  # equal-size chunks only
    for c in chunks[:warmup]:
        predict(c)
    times = []
    for c in chunks:
        _sync(device)
        t = time.perf_counter()
        predict(c)
        _sync(device)
        times.append((time.perf_counter() - t) * 1000)
    t_arr = np.array(times)
    n = sum(len(c) for c in chunks)
    return {"batch_size": batch_size, "device": device, "requests": len(chunks),
            "ms_per_batch_p50": float(np.median(t_arr)), "ms_per_batch_p95": float(np.percentile(t_arr, 95)),
            "ms_per_text": float(t_arr.sum() / n), "texts_per_s": float(n / (t_arr.sum() / 1000))}
