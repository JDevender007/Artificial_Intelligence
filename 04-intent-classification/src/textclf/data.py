"""Banking77 download, loading, the fixed stratified train/val split and computed dataset statistics."""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]  # index -> intent name (sorted, so ids are stable across runs)


def download(url: str, dest: str | Path, retries: int = 5, timeout: int = 300) -> Path:
    """Download ``url`` to ``dest`` unless it already exists (with retries for flaky networks)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
                dest.write_bytes(r.read())
            log.info("downloaded %s (%d bytes)", dest, dest.stat().st_size)
            return dest
        except OSError as e:
            log.warning("download %s failed (%d/%d): %s", url, attempt, retries, e)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {url}; fetch it manually into {dest}")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a Banking77 CSV (``text,category``) into columns ``text, intent`` with whitespace cleaned."""
    df = pd.read_csv(path)
    if not {"text", "category"} <= set(df.columns):
        raise ValueError(f"{path}: expected columns text, category; got {list(df.columns)}")
    df = df.rename(columns={"category": "intent"})[["text", "intent"]]
    df["text"] = df["text"].astype(str).str.split().str.join(" ")
    return df.reset_index(drop=True)


def word_count(texts: pd.Series | list[str]) -> np.ndarray:
    return np.array([len(str(t).split()) for t in texts])


def make_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: DataConfig) -> Splits:
    """Stratified val split from the official train set; the official test set stays untouched."""
    labels = sorted(train_df["intent"].unique())
    missing = set(test_df["intent"]) - set(labels)
    if missing:
        raise ValueError(f"test intents not in train: {sorted(missing)}")
    lab = {name: i for i, name in enumerate(labels)}
    train_df, test_df = train_df.assign(label=train_df.intent.map(lab)), test_df.assign(label=test_df.intent.map(lab))
    tr, va = train_test_split(train_df, test_size=cfg.val_fraction, random_state=cfg.split_seed,
                              stratify=train_df["label"])
    if cfg.max_train is not None and cfg.max_train < len(tr):
        tr, _ = train_test_split(tr, train_size=cfg.max_train, random_state=cfg.split_seed, stratify=tr["label"])
    return Splits(tr.reset_index(drop=True), va.reset_index(drop=True), test_df.reset_index(drop=True), labels)


def load_splits(cfg: DataConfig) -> Splits:
    raw = Path(cfg.raw_dir)
    train_df = load_csv(download(cfg.train_url, raw / "train.csv"))
    test_df = load_csv(download(cfg.test_url, raw / "test.csv"))
    return make_splits(train_df, test_df, cfg)


def split_stats(s: Splits) -> dict:
    """Rows, classes, per-class count range, word-length quantiles and duplicate/overlap checks."""
    out: dict = {"n_classes": len(s.labels)}
    for name, df in (("train", s.train), ("val", s.val), ("test", s.test)):
        counts = df["intent"].value_counts()
        wc = word_count(df["text"])
        out[name] = {"rows": len(df), "classes": int(counts.size), "min_per_class": int(counts.min()),
                     "max_per_class": int(counts.max()), "words_mean": float(wc.mean()),
                     "words_p50": float(np.median(wc)), "words_p95": float(np.percentile(wc, 95)),
                     "words_max": int(wc.max())}
    train_texts = set(s.train.text.str.lower())
    out["test_exact_overlap_with_train"] = int(s.test.text.str.lower().isin(train_texts).sum())
    out["val_exact_overlap_with_train"] = int(s.val.text.str.lower().isin(train_texts).sum())
    return out


def stats_markdown(st: dict) -> str:
    lines = [f"{st['n_classes']} intents", "",
             "| split | rows | classes | per class (min-max) | words mean | p50 | p95 | max |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("train", "val", "test"):
        r = st[name]
        lines.append(f"| {name} | {r['rows']} | {r['classes']} | {r['min_per_class']}-{r['max_per_class']} | "
                     f"{r['words_mean']:.1f} | {r['words_p50']:.0f} | {r['words_p95']:.0f} | {r['words_max']} |")
    lines += ["", f"Exact (case-insensitive) duplicates of a train text: test {st['test_exact_overlap_with_train']}, "
                  f"val {st['val_exact_overlap_with_train']}."]
    return "\n".join(lines)
