"""Popularity, BPR matrix factorisation and a feature two-tower model with mixed negative sampling.

Every model exposes ``score_fn(users) -> (len(users), n_items)`` for evaluation at a given cutoff:
``"val"`` (history = train) or ``"test"`` (history = train + val). Models are only ever fit on the train split.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import BPRConfig, TowerConfig, TrainConfig
from .data import GENRES, N_OCC, N_YEAR, Dataset
from .metrics import ScoreFn
from .utils import get_logger

log = get_logger()
CUTOFF_SPLITS = {"val": ("train",), "test": ("train", "val")}


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def known_users(ds: Dataset) -> np.ndarray:
    """Users with at least one rating in train (everyone else is a cold-start user)."""
    m = np.zeros(ds.n_users, dtype=bool)
    m[ds.part("train")["u"].unique()] = True
    return m


def known_items(ds: Dataset) -> np.ndarray:
    m = np.zeros(ds.n_items, dtype=bool)
    m[ds.part("train")["i"].unique()] = True
    return m


# --------------------------------------------------------------------------- popularity
class Popularity:
    """Most-liked items in train, the same list for every user (minus what they have seen)."""

    def __init__(self, ds: Dataset) -> None:
        self.scores = np.log1p(ds.item_counts("train")).astype(np.float32)

    def score_fn(self, cutoff: str) -> ScoreFn:
        return lambda users: np.broadcast_to(self.scores, (len(users), len(self.scores)))


# --------------------------------------------------------------------------- training loop
def fit(model: nn.Module, batches: Callable[[], Iterator[torch.Tensor]], val_metric: Callable[[], float],
        cfg: TrainConfig, lr: float, weight_decay: float = 0.0) -> dict[str, Any]:
    """Adam with early stopping on ``val_metric`` (higher is better); restores the best epoch's weights."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for loss in batches():
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        model.eval()
        with torch.no_grad():
            v = val_metric()
        history.append({"epoch": epoch, "loss": total / max(n, 1), "val": v})
        log.info("epoch %d loss %.4f val %.4f", epoch, total / max(n, 1), v)
        if v > best:
            best, best_state, best_epoch, bad = v, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_val": best, "best_epoch": best_epoch, "epochs_run": len(history),
            "train_seconds": time.perf_counter() - t0, "history": history}


# --------------------------------------------------------------------------- BPR-MF
class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        nn.init.zeros_(self.bias.weight)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user(u) * self.item(i)).sum(-1) + self.bias(i).squeeze(-1)

    def full_scores(self, u: torch.Tensor) -> torch.Tensor:
        return self.user(u) @ self.item.weight.T + self.bias.weight.T


class BPRRecommender:
    """BPR with uniform negatives. Users with no train ratings fall back to popularity (MF cannot embed them)."""

    def __init__(self, ds: Dataset, cfg: BPRConfig, train_cfg: TrainConfig, device: torch.device) -> None:
        self.ds, self.cfg, self.train_cfg, self.device = ds, cfg, train_cfg, device
        self.model = BPRMF(ds.n_users, ds.n_items, cfg.dim).to(device)
        self.known = known_users(ds)
        self.pop = Popularity(ds).scores

    def _batches(self) -> Iterator[torch.Tensor]:
        p = self.ds.part("train", positives=True)
        u_all = torch.as_tensor(p["u"].to_numpy(), device=self.device)
        i_all = torch.as_tensor(p["i"].to_numpy(), device=self.device)
        perm = torch.randperm(len(u_all), device=self.device)
        bs = self.train_cfg.batch_size
        for s in range(0, len(perm), bs):
            idx = perm[s: s + bs]
            u, i = u_all[idx], i_all[idx]
            j = torch.randint(0, self.ds.n_items, (len(idx),), device=self.device)
            x = self.model.score(u, i) - self.model.score(u, j)
            reg = (self.model.user(u).pow(2) + self.model.item(i).pow(2) + self.model.item(j).pow(2)).sum(-1)
            yield -F.logsigmoid(x).mean() + self.cfg.reg * reg.mean()

    def fit(self, val_metric: Callable[[ScoreFn], float]) -> dict[str, Any]:
        return fit(self.model, self._batches, lambda: val_metric(self.score_fn("val")), self.train_cfg,
                   self.cfg.lr)

    def score_fn(self, cutoff: str) -> ScoreFn:
        def f(users: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                s = self.model.full_scores(torch.as_tensor(users, device=self.device)).float().cpu().numpy()
            cold = ~self.known[users]
            s[cold] = self.pop  # documented fallback for cold-start users
            return s
        return f


# --------------------------------------------------------------------------- two-tower
class TwoTower(nn.Module):
    """User tower: id + gender/age/occupation + mean of recent liked items. Item tower: id + genres + decade.

    Index ``n_users`` / ``n_items`` of the id tables means "unknown": ids are dropped to it during training
    (``id_dropout``) and cold users/items always use it, so they are embedded from their features alone.
    """

    def __init__(self, n_users: int, n_items: int, cfg: TowerConfig, item_genres: np.ndarray,
                 item_year: np.ndarray) -> None:
        super().__init__()
        self.n_users, self.n_items, self.cfg = n_users, n_items, cfg
        e = cfg.emb
        self.user_id = nn.Embedding(n_users + 1, e)
        self.hist = nn.EmbeddingBag(n_items + 1, e, mode="mean", padding_idx=n_items)
        self.item_id = nn.Embedding(n_items + 1, e)
        u_in, i_in = 2 * e, e
        if cfg.use_features:
            self.gender, self.age, self.occ = nn.Embedding(2, e), nn.Embedding(7, e), nn.Embedding(N_OCC, e)
            self.genre = nn.EmbeddingBag(len(GENRES) + 1, e, mode="mean", padding_idx=len(GENRES))
            self.year = nn.Embedding(N_YEAR, e)
            u_in, i_in = u_in + 3 * e, i_in + 2 * e
        self.user_mlp = nn.Sequential(nn.Linear(u_in, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.dim))
        self.item_mlp = nn.Sequential(nn.Linear(i_in, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, cfg.dim))
        self.register_buffer("item_genres", torch.as_tensor(item_genres, dtype=torch.long))
        self.register_buffer("item_year", torch.as_tensor(item_year, dtype=torch.long))
        for emb in (self.user_id, self.item_id):
            nn.init.normal_(emb.weight, std=0.05)

    def user_vec(self, uid: torch.Tensor, feat: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        h = torch.where(hist < 0, torch.full_like(hist, self.n_items), hist)
        parts = [self.user_id(uid), self.hist(h)]
        if self.cfg.use_features:
            parts += [self.gender(feat[:, 0]), self.age(feat[:, 1]), self.occ(feat[:, 2])]
        return F.normalize(self.user_mlp(torch.cat(parts, -1)), dim=-1)

    def item_vec(self, iid: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """``iid`` is the id fed to the id table (maybe "unknown"); ``items`` selects the features."""
        parts = [self.item_id(iid)]
        if self.cfg.use_features:
            parts += [self.genre(self.item_genres[items]), self.year(self.item_year[items])]
        return F.normalize(self.item_mlp(torch.cat(parts, -1)), dim=-1)


def causal_histories(ds: Dataset, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train positives as (u, i, history) where history = the user's previous ``length`` train positives."""
    p = ds.part("train", positives=True)
    us, its, hs = [], [], []
    for u, g in p.groupby("u", sort=False):
        items = g["i"].to_numpy()
        pad = np.concatenate([np.full(length, -1, dtype=np.int64), items])
        win = np.lib.stride_tricks.sliding_window_view(pad, length)[: len(items)]
        us.append(np.full(len(items), u, dtype=np.int64))
        its.append(items)
        hs.append(win)
    if not us:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, length), np.int64)
    return np.concatenate(us), np.concatenate(its), np.concatenate(hs)


class TwoTowerRecommender:
    """Sampled softmax over in-batch positives (optionally logQ-corrected) plus shared uniform negatives."""

    def __init__(self, ds: Dataset, cfg: TowerConfig, train_cfg: TrainConfig, device: torch.device) -> None:
        self.ds, self.cfg, self.train_cfg, self.device = ds, cfg, train_cfg, device
        self.model = TwoTower(ds.n_users, ds.n_items, cfg, ds.item_genres, ds.item_year).to(device)
        self.known_u = torch.as_tensor(known_users(ds), device=device)
        self.known_i = torch.as_tensor(known_items(ds), device=device)
        self.user_feat = torch.as_tensor(ds.user_feat, device=device)
        counts = ds.item_counts("train").astype(np.float64) + 1.0
        self.log_q = torch.as_tensor(np.log(counts / counts.sum()), dtype=torch.float32, device=device)
        u, i, h = causal_histories(ds, cfg.history_len)
        self.tr_u = torch.as_tensor(u, device=device)
        self.tr_i = torch.as_tensor(i, device=device)
        self.tr_h = torch.as_tensor(h, device=device)

    def _drop(self, ids: torch.Tensor, unknown: int) -> torch.Tensor:
        if self.cfg.id_dropout <= 0:
            return ids
        return torch.where(torch.rand(ids.shape, device=ids.device) < self.cfg.id_dropout,
                           torch.full_like(ids, unknown), ids)

    def _batches(self) -> Iterator[torch.Tensor]:
        m, c, dev = self.model, self.cfg, self.device
        n_items = self.ds.n_items
        perm = torch.randperm(len(self.tr_u), device=dev)
        for s in range(0, len(perm), self.train_cfg.batch_size):
            idx = perm[s: s + self.train_cfg.batch_size]
            u, i, h = self.tr_u[idx], self.tr_i[idx], self.tr_h[idx]
            uv = m.user_vec(self._drop(u, m.n_users), self.user_feat[u], h)
            iv = m.item_vec(self._drop(i, n_items), i)
            logits = uv @ iv.T / c.temperature
            if c.logq:
                logits = logits - (self.log_q[i] + np.log(len(idx)))[None, :]
            same = i[:, None] == i[None, :]
            same.fill_diagonal_(False)  # the same item as another row's positive is not a negative
            logits = logits.masked_fill(same, float("-inf"))
            if c.n_random_neg > 0:
                r = torch.randint(0, n_items, (c.n_random_neg,), device=dev)
                rl = uv @ m.item_vec(self._drop(r, n_items), r).T / c.temperature
                if c.logq:
                    rl = rl - float(np.log(c.n_random_neg / n_items))
                rl = rl.masked_fill(r[None, :] == i[:, None], float("-inf"))
                logits = torch.cat([logits, rl], 1)
            yield F.cross_entropy(logits, torch.arange(len(idx), device=dev))

    def fit(self, val_metric: Callable[[ScoreFn], float]) -> dict[str, Any]:
        return fit(self.model, self._batches, lambda: val_metric(self.score_fn("val")), self.train_cfg,
                   self.cfg.lr, self.cfg.weight_decay)

    @torch.no_grad()
    def item_vectors(self) -> torch.Tensor:
        items = torch.arange(self.ds.n_items, device=self.device)
        iid = torch.where(self.known_i, items, torch.full_like(items, self.ds.n_items))
        return self.model.item_vec(iid, items)

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray, cutoff: str) -> torch.Tensor:
        hist = self.ds.history(users, *CUTOFF_SPLITS[cutoff], length=self.cfg.history_len)
        u = torch.as_tensor(users, device=self.device)
        uid = torch.where(self.known_u[u], u, torch.full_like(u, self.ds.n_users))
        return self.model.user_vec(uid, self.user_feat[u], torch.as_tensor(hist, device=self.device))

    def score_fn(self, cutoff: str) -> ScoreFn:
        items = self.item_vectors()

        def f(users: np.ndarray) -> np.ndarray:
            return (self.user_vectors(users, cutoff) @ items.T).float().cpu().numpy()
        return f

    def checkpoint(self) -> dict[str, Any]:
        """Everything the API needs to embed a new user from features + liked movies."""
        return {"state": self.model.state_dict(), "n_users": self.ds.n_users, "n_items": self.ds.n_items,
                "tower": asdict(self.cfg)}


def load_two_tower(ckpt: dict[str, Any], device: str = "cpu") -> TwoTower:
    cfg = TowerConfig(**ckpt["tower"])
    st = ckpt["state"]
    m = TwoTower(ckpt["n_users"], ckpt["n_items"], cfg, st["item_genres"].cpu().numpy(),
                 st["item_year"].cpu().numpy())
    m.load_state_dict(st)
    return m.to(device).eval()
