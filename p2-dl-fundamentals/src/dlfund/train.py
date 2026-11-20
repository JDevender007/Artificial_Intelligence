"""Hand-written training loop (no Trainer / Lightning).

Features: SGD/AdamW without weight decay on norm/bias params, OneCycle or cosine LR
stepped per batch, mixed precision via ``torch.amp`` (``torch.cuda.amp.*`` is
deprecated since PyTorch 2.4), optional gradient clipping, early stopping on val loss,
TensorBoard scalars, and a ``last.pt`` checkpoint every epoch that a crashed run resumes from.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dlfund.config import TrainConfig


class EarlyStopping:
    """Stop when the monitored value has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record ``value`` (lower is better); return True if it is a new best."""
        if value < self.best - self.min_delta:
            self.best, self.bad_epochs = value, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, float]:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, s: dict[str, float]) -> None:
        self.best, self.bad_epochs = s["best"], int(s["bad_epochs"])


def build_optimizer(model: nn.Module, t: TrainConfig) -> torch.optim.Optimizer:
    """Weight decay only on weight matrices/kernels, never on biases or norm scales."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": t.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if t.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=t.lr, momentum=t.momentum, nesterov=True)
    return torch.optim.AdamW(groups, lr=t.lr)


def build_scheduler(opt: torch.optim.Optimizer, t: TrainConfig, steps_per_epoch: int) -> Any:
    """Per-batch LR schedule, or None."""
    total = t.epochs * steps_per_epoch
    if t.scheduler == "onecycle":
        if total < 4:  # OneCycle needs >= 2 steps per phase; tiny runs (tests) use a constant LR
            return None
        # Warm up for 25% of training, but at least 2 steps (PyTorch divides by the phase length).
        pct = max(0.25, 2 / total)
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t.lr, total_steps=total, pct_start=pct)
    if t.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    return None


def _autocast(device: torch.device, amp: bool) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=amp and device.type == "cuda")


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler, sched: Any,
                    device: torch.device, amp: bool, grad_clip: float | None) -> tuple[float, float, int]:
    """One pass over ``loader``. Returns ``(mean loss, accuracy, n_samples)``."""
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        bs = y.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return loss_sum / n, correct / n, n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            amp: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(mean cross-entropy, logits [N, C] float32, targets [N])``."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        with _autocast(device, amp):
            logits = model(x.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    logits_t, y_t = torch.cat(all_logits), torch.cat(all_y)
    loss = nn.functional.cross_entropy(logits_t, y_t).item()
    return loss, logits_t.numpy(), y_t.numpy()


@dataclass
class FitResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    epochs_run: int = 0
    train_seconds: float = 0.0
    images_per_s: float = 0.0
    stopped_early: bool = False


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, t: TrainConfig,
        device: torch.device, ckpt_dir: Path, writer: Any = None,
        logger: Logger | None = None) -> FitResult:
    """Train with early stopping; resume from ``ckpt_dir/last.pt``; leave best weights loaded."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=t.amp and device.type == "cuda")
    stopper = EarlyStopping(t.early_stop_patience)
    res = FitResult()
    start_epoch = 0

    if last_path.exists():  # resume after a crash / Kaggle timeout
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck["sched"] is not None:
            sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        stopper.load_state_dict(ck["stopper"])
        res = FitResult(**ck["result"])
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["rng_cpu"])
        if device.type == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        if logger:
            logger.info("Resumed from %s at epoch %d", last_path, start_epoch)

    for epoch in range(start_epoch, t.epochs):
        if stopper.should_stop:
            break
        t0 = time.perf_counter()
        tr_loss, tr_acc, n = train_one_epoch(model, train_loader, criterion, opt, scaler, sched,
                                             device, t.amp, t.grad_clip)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        va_loss, va_logits, va_y = predict(model, val_loader, device, t.amp)
        va_acc = float((va_logits.argmax(1) == va_y).mean())
        lr = opt.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
               "val_acc": va_acc, "lr": lr, "epoch_seconds": dt, "images_per_s": n / dt}
        res.history.append(row)
        res.epochs_run = epoch + 1
        res.train_seconds += dt
        if stopper.step(va_loss):
            res.best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
        if writer is not None:
            writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch + 1)
            writer.add_scalars("acc", {"train": tr_acc, "val": va_acc}, epoch + 1)
            writer.add_scalar("lr", lr, epoch + 1)
        if logger:
            logger.info("  epoch %3d  train %.4f/%.4f  val %.4f/%.4f  lr %.2e  %.0f img/s",
                        epoch + 1, tr_loss, tr_acc, va_loss, va_acc, lr, n / dt)
        res.stopped_early = stopper.should_stop and epoch + 1 < t.epochs
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "scaler": scaler.state_dict(), "stopper": stopper.state_dict(),
            "result": res.__dict__, "epoch": epoch, "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        }, last_path)

    res.images_per_s = float(np.mean([h["images_per_s"] for h in res.history])) if res.history else 0.0
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return res
