"""Neural forecasters (PyTorch): an LSTM and a small Transformer encoder, both direct multi-output.

Both are *global* models: trained once on the training period (early stopping on val), then applied at every
backtest origin to the last ``lookback`` hours without re-training. Inputs per hour: the standardised load plus
sin/cos of hour-of-day and day-of-week. Output: the next ``horizon`` hours at once (no autoregressive roll-out,
so errors do not compound across steps).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import NNConfig
from .utils import get_logger, set_seed

log = get_logger(__name__)
N_FEATURES = 5


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour.to_numpy() / 24.0
    d = index.dayofweek.to_numpy() / 7.0
    return np.stack([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h), np.sin(2 * np.pi * d), np.cos(2 * np.pi * d)], 1)


class LSTMNet(nn.Module):
    def __init__(self, cfg: NNConfig, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, cfg.hidden, cfg.layers, batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0)
        self.head = nn.Linear(cfg.hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TransformerNet(nn.Module):
    """Pre-norm encoder over the lookback window; the head reads the last position and the mean of all positions."""

    def __init__(self, cfg: NNConfig, horizon: int):
        super().__init__()
        d = cfg.hidden
        self.inp = nn.Linear(N_FEATURES, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.lookback, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(2 * d, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.enc(self.inp(x) + self.pos[:, -x.shape[1] :]))
        return self.head(torch.cat([h[:, -1], h.mean(1)], dim=-1))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class NeuralForecaster:
    deterministic = False

    def __init__(self, kind: str, cfg: NNConfig, horizon: int, seed: int):
        if kind not in ("lstm", "transformer"):
            raise ValueError(kind)
        self.name, self.cfg, self.horizon, self.seed = kind, cfg, horizon, seed
        self.device = _device(cfg.device)
        self.net: nn.Module | None = None
        self.mu = self.sd = 0.0

    def _features(self, y: pd.Series) -> np.ndarray:
        z = (y.to_numpy(dtype=np.float64) - self.mu) / self.sd
        return np.concatenate([z[:, None], calendar_features(y.index)], 1).astype(np.float32)

    def _windows(self, F: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L, H = self.cfg.lookback, self.horizon
        xi = t[:, None] + torch.arange(-L, 0, device=t.device)
        yi = t[:, None] + torch.arange(0, H, device=t.device)
        return F[xi], z[yi]

    def fit(self, train: pd.Series, val: pd.Series) -> dict:
        set_seed(self.seed)
        cfg, L, H = self.cfg, self.cfg.lookback, self.horizon
        self.mu, self.sd = float(train.mean()), float(train.std())
        full = pd.concat([train, val])
        F = torch.from_numpy(self._features(full)).to(self.device)
        z = F[:, 0].contiguous()
        n_tr = len(train)
        t_train = torch.arange(L, n_tr - H + 1, device=self.device)  # targets end inside train
        t_val = torch.arange(max(n_tr, L), len(full) - H + 1, 6, device=self.device)  # targets inside val
        if len(t_train) == 0 or len(t_val) == 0:
            raise ValueError("train or val too short for lookback + horizon")
        torch.manual_seed(self.seed)
        net = (LSTMNet if self.name == "lstm" else TransformerNet)(cfg, H).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.L1Loss()  # the headline metric is MAE, and L1 targets the conditional median
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        best, best_state, bad, hist = math.inf, None, 0, []
        t0 = time.perf_counter()
        for ep in range(cfg.epochs):
            net.train()
            perm = t_train[torch.randperm(len(t_train), generator=g).to(self.device)]
            tot = 0.0
            for i in range(0, len(perm), cfg.batch_size):
                xb, yb = self._windows(F, z, perm[i : i + cfg.batch_size])
                loss = loss_fn(net(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(xb)
            val_mae = self._eval(net, F, z, t_val) * self.sd  # in kW
            hist.append({"epoch": ep + 1, "train_l1": tot / len(perm), "val_mae_kw": val_mae})
            log.info("%s seed %d epoch %d: train L1 %.4f, val MAE %.4f kW", self.name, self.seed, ep + 1, tot / len(perm), val_mae)
            if val_mae < best - 1e-5:
                best, best_state, bad = val_mae, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        best_ep = min(hist, key=lambda r: r["val_mae_kw"])["epoch"]
        return {
            "params": n_params(net),
            "epochs_run": len(hist),
            "best_epoch": best_ep,
            "val_mae_kw": best,
            "train_s": time.perf_counter() - t0,
            "device": str(self.device),
            "history": hist,
        }

    @torch.no_grad()
    def _eval(self, net: nn.Module, F: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> float:
        net.eval()
        errs = []
        for i in range(0, len(t), 1024):
            xb, yb = self._windows(F, z, t[i : i + 1024])
            errs.append((net(xb) - yb).abs().mean(1))
        return float(torch.cat(errs).mean())

    def save(self, path) -> None:
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd}, path)

    def load(self, path) -> NeuralForecaster:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.net = (LSTMNet if self.name == "lstm" else TransformerNet)(self.cfg, self.horizon).to(self.device)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.mu, self.sd = ck["mu"], ck["sd"]
        return self

    @torch.no_grad()
    def predict(self, hist: pd.Series, horizon: int) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("fit() first")
        if horizon != self.horizon:
            raise ValueError(f"trained for horizon {self.horizon}, asked for {horizon}")
        x = torch.from_numpy(self._features(hist.iloc[-self.cfg.lookback :])).to(self.device)[None]
        return self.net(x)[0].cpu().numpy().astype(np.float64) * self.sd + self.mu
