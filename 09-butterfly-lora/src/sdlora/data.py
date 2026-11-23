"""Smithsonian butterflies (CC0-1.0): download, caption, split and square-pad the 1000 specimen photos.

Source: ``huggan/smithsonian_butterflies_subset`` (1000 images, one per species, background removed), a subset of
``ceyda/smithsonian_butterflies`` whose card states CC0-1.0 (Smithsonian Open Access). The revision is pinned so the
split is reproducible. Images are never committed; they are rebuilt under ``<output_dir>/data``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from sdlora.utils import get_logger, read_jsonl

log = get_logger(__name__)

REPO_ID = "huggan/smithsonian_butterflies_subset"
REVISION = "3cdedf844922ab40393d46d4c7f81c596e1c6d45"
PARQUET = "data/train-00000-of-00001.parquet"
CAPTION = "a museum specimen photo of a {name}, {family} family"


def family_of(taxonomy: Any) -> str:
    """The ``...idae`` rank of a comma-separated taxonomy string, else ``Lepidoptera``."""
    if isinstance(taxonomy, str):
        for part in (p.strip() for p in taxonomy.split(",")):
            if part.endswith("idae") and part[:1].isupper():
                return part
    return "Lepidoptera"


def clean_name(name: Any, scientific: Any) -> str:
    for v in (name, scientific):
        if isinstance(v, str) and v.strip():
            return " ".join(v.split())
    return "unidentified butterfly"


def caption_for(name: Any, scientific: Any, taxonomy: Any) -> str:
    return CAPTION.format(name=clean_name(name, scientific), family=family_of(taxonomy))


def split_indices(n: int, n_test: int, seed: int) -> np.ndarray:
    """Boolean mask, True = test. Fixed by ``seed`` so every run and model sees the same held-out species."""
    if not 0 < n_test < n:
        raise ValueError(f"n_test must be in (0, {n})")
    mask = np.zeros(n, dtype=bool)
    mask[np.random.default_rng(seed).permutation(n)[:n_test]] = True
    return mask


def to_square(img: Image.Image, size: int) -> Image.Image:
    """Pad to a square with the median border colour (keeps the full wingspan), then resize."""
    img = img.convert("RGB")
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = tuple(int(c) for c in np.median(border, axis=0))
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), bg)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def _image_bytes(cell: Any) -> bytes:
    if isinstance(cell, dict):
        return cell["bytes"]
    if isinstance(cell, bytes | bytearray):
        return bytes(cell)
    raise TypeError(f"unexpected image cell {type(cell)}")


def records_from_frame(df: pd.DataFrame, n_test: int, split_seed: int) -> list[dict[str, Any]]:
    """Metadata rows (no pixels) in dataset order, with caption and split."""
    test = split_indices(len(df), n_test, split_seed)
    recs = []
    for i, r in enumerate(df.itertuples(index=False)):
        recs.append({
            "idx": i, "id": r.id, "name": clean_name(r.name, r.scientific_name),
            "scientific_name": r.scientific_name if isinstance(r.scientific_name, str) else "",
            "family": family_of(r.taxonomy), "caption": caption_for(r.name, r.scientific_name, r.taxonomy),
            "split": "test" if test[i] else "train", "source": r.source if isinstance(r.source, str) else "",
        })
    return recs


def write_images(df: pd.DataFrame, records: list[dict[str, Any]], image_dir: Path, size: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for rec, cell in zip(records, df["image"], strict=True):
        p = image_dir / f"{rec['idx']:04d}.png"
        if not p.exists():
            to_square(Image.open(io.BytesIO(_image_bytes(cell))), size).save(p)


def prepare_dataset(data_dir: str | Path, n_test: int = 200, split_seed: int = 0, resolution: int = 512,
                    limit: int = 0) -> list[dict[str, Any]]:
    """Download (once), caption, split and pad. ``limit`` > 0 keeps the first rows (smoke runs only)."""
    data_dir = Path(data_dir)
    meta = data_dir / "metadata.jsonl"
    if meta.exists():
        return read_jsonl(meta)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    recs = records_from_frame(df, n_test, split_seed)
    log.info("dataset: %d images (%d test), writing %dpx squares", len(recs), n_test, resolution)
    write_images(df, recs, data_dir / "images", resolution)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return recs


def image_path(data_dir: str | Path, rec: dict[str, Any]) -> Path:
    return Path(data_dir) / "images" / f"{rec['idx']:04d}.png"
