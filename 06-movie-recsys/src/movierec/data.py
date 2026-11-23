"""MovieLens-1M: download, parse, side features and a global temporal split.

Positives are ratings >= ``min_rating``; every rating (any value) counts as "seen" and is excluded from ranking.
The split is global by time: train = ratings before the ``1 - val - test`` timestamp quantile, val = the next
``val_frac``, test = the last ``test_frac``. No rating from the future is used to fit a model or to build history.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DataConfig
from .utils import get_logger

log = get_logger()

GENRES = (
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)
AGES = (1, 18, 25, 35, 45, 50, 56)  # ML-1M age codes ("1" = under 18, "56" = 56+)
N_OCC = 21
N_YEAR = 11  # decades 1910s..2000s -> 0..9 (earlier clipped to 0), 10 = unknown
MAX_GENRES = 6
SPLITS = ("train", "val", "test")


@dataclass
class Dataset:
    """Contiguous indices: users 0..n_users-1, items 0..n_items-1 (raw ids kept for display and the API)."""

    ratings: pd.DataFrame  # u, i, rating, ts, pos (bool), split, sorted by (ts, u, i)
    user_raw: np.ndarray
    item_raw: np.ndarray
    titles: list[str]
    genres: list[str]  # raw "A|B|C" strings
    user_feat: np.ndarray  # (n_users, 3): gender, age bucket, occupation
    item_genres: np.ndarray  # (n_items, MAX_GENRES), padded with len(GENRES)
    item_year: np.ndarray  # (n_items,) decade bucket
    t_val: int
    t_test: int

    @property
    def n_users(self) -> int:
        return len(self.user_raw)

    @property
    def n_items(self) -> int:
        return len(self.item_raw)

    def part(self, *splits: str, positives: bool = False) -> pd.DataFrame:
        r = self.ratings[self.ratings["split"].isin(splits)]
        return r[r["pos"]] if positives else r

    def seen_matrix(self, *splits: str) -> sp.csr_matrix:
        """Users x items, 1 where the user rated the item in ``splits`` (excluded from ranking)."""
        r = self.part(*splits)
        return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r["u"], r["i"])),
                             shape=(self.n_users, self.n_items))

    def targets(self, split: str) -> dict[int, np.ndarray]:
        """Positive items per user in ``split`` (only users with at least one)."""
        p = self.part(split, positives=True)
        return {int(u): g["i"].to_numpy() for u, g in p.groupby("u", sort=True)}

    def item_counts(self, *splits: str) -> np.ndarray:
        """Number of positives per item in ``splits``."""
        p = self.part(*splits, positives=True)
        return np.bincount(p["i"].to_numpy(), minlength=self.n_items).astype(np.int64)

    def history(self, users: np.ndarray, *splits: str, length: int) -> np.ndarray:
        """(len(users), length) last positives in ``splits`` per user, oldest first, left-padded with -1."""
        p = self.part(*splits, positives=True)
        out = np.full((len(users), length), -1, dtype=np.int64)
        groups = {int(u): g["i"].to_numpy() for u, g in p.groupby("u", sort=False)}  # rows already time-sorted
        for row, u in enumerate(users):
            h = groups.get(int(u))
            if h is not None and len(h):
                h = h[-length:]
                out[row, length - len(h):] = h
        return out


def _is_complete_zip(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def download(cfg: DataConfig) -> Path:
    """Path to ml-1m.zip: reuse a complete file, else write synthetic data or download it."""
    raw = Path(cfg.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if cfg.synthetic:
        path = raw / f"synthetic_{cfg.synthetic_users}x{cfg.synthetic_items}.zip"
        if not _is_complete_zip(path):
            write_synthetic_zip(path, cfg.synthetic_users, cfg.synthetic_items)
        return path
    path = raw / "ml-1m.zip"
    if _is_complete_zip(path):  # a cut-off download has no central directory
        return path
    log.info("downloading %s", cfg.url)
    part = path.with_suffix(".part")
    urllib.request.urlretrieve(cfg.url, part)
    if not zipfile.is_zipfile(part):
        raise OSError(f"{cfg.url} did not give a complete zip; delete {part} and retry")
    part.replace(path)
    return path


def _read_dat(zf: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    member = next(n for n in zf.namelist() if n.endswith(name))
    text = zf.read(member).decode("latin-1")  # ML-1M titles are ISO-8859-1
    rows = [line.split("::") for line in text.splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=cols)


def load_raw(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as zf:
        ratings = _read_dat(zf, "ratings.dat", ["user", "item", "rating", "ts"])
        users = _read_dat(zf, "users.dat", ["user", "gender", "age", "occupation", "zip"])
        movies = _read_dat(zf, "movies.dat", ["item", "title", "genres"])
    for c in ratings.columns:
        ratings[c] = ratings[c].astype(np.int64)
    users["user"] = users["user"].astype(np.int64)
    movies["item"] = movies["item"].astype(np.int64)
    return ratings, users, movies


def _year_bucket(title: str) -> int:
    m = re.search(r"\((\d{4})\)\s*$", title)
    if not m:
        return N_YEAR - 1
    return int(np.clip((int(m.group(1)) - 1910) // 10, 0, N_YEAR - 2))


def build_dataset(cfg: DataConfig) -> Dataset:
    ratings, users, movies = load_raw(download(cfg))
    users = users.sort_values("user").reset_index(drop=True)
    movies = movies.sort_values("item").reset_index(drop=True)
    u_index = pd.Series(np.arange(len(users)), index=users["user"])
    i_index = pd.Series(np.arange(len(movies)), index=movies["item"])

    age_idx = {a: k for k, a in enumerate(AGES)}
    user_feat = np.stack([
        (users["gender"] == "M").astype(np.int64).to_numpy(),
        users["age"].astype(int).map(age_idx).to_numpy(np.int64),
        users["occupation"].astype(np.int64).to_numpy(),
    ], axis=1)
    g_idx = {g: k for k, g in enumerate(GENRES)}
    item_genres = np.full((len(movies), MAX_GENRES), len(GENRES), dtype=np.int64)
    for row, gs in enumerate(movies["genres"]):
        ids = [g_idx[g] for g in gs.split("|") if g in g_idx][:MAX_GENRES]
        item_genres[row, : len(ids)] = ids
    item_year = movies["title"].map(_year_bucket).to_numpy(np.int64)

    r = pd.DataFrame({
        "u": u_index.loc[ratings["user"]].to_numpy(),
        "i": i_index.loc[ratings["item"]].to_numpy(),
        "rating": ratings["rating"].to_numpy(),
        "ts": ratings["ts"].to_numpy(),
    }).sort_values(["ts", "u", "i"], kind="stable").reset_index(drop=True)
    r["pos"] = r["rating"] >= cfg.min_rating
    t_val = int(np.quantile(r["ts"], 1 - cfg.val_frac - cfg.test_frac))
    t_test = int(np.quantile(r["ts"], 1 - cfg.test_frac))
    r["split"] = np.where(r["ts"] < t_val, "train", np.where(r["ts"] < t_test, "val", "test"))

    return Dataset(ratings=r, user_raw=users["user"].to_numpy(), item_raw=movies["item"].to_numpy(),
                   titles=movies["title"].tolist(), genres=movies["genres"].tolist(), user_feat=user_feat,
                   item_genres=item_genres, item_year=item_year, t_val=t_val, t_test=t_test)


def summary(ds: Dataset) -> dict[str, float | int | str]:
    """Counts reported in data_summary.md (all computed, nothing typed in by hand)."""
    out: dict[str, float | int | str] = {
        "users": ds.n_users, "items": ds.n_items, "ratings": len(ds.ratings),
        "positives": int(ds.ratings["pos"].sum()),
        "density": len(ds.ratings) / (ds.n_users * ds.n_items),
        "t_val": pd.to_datetime(ds.t_val, unit="s").isoformat(),
        "t_test": pd.to_datetime(ds.t_test, unit="s").isoformat(),
    }
    train_users = set(ds.part("train")["u"])
    train_items = set(ds.part("train")["i"])
    for s in SPLITS:
        part = ds.part(s)
        pos = part[part["pos"]]
        out[f"{s}_ratings"] = len(part)
        out[f"{s}_positives"] = len(pos)
        out[f"{s}_users_with_pos"] = pos["u"].nunique()
        if s != "train":
            out[f"{s}_cold_users"] = len(set(pos["u"]) - train_users)
            out[f"{s}_cold_item_positives"] = int((~pos["i"].isin(train_items)).sum())
    return out


def write_synthetic_zip(path: str | Path, n_users: int, n_items: int, seed: int = 0) -> None:
    """A small fake ml-1m.zip in the real file format, with learnable structure.

    Users prefer two genres (drawn partly from their gender/age), rate popular items more often, and give
    4-5 stars to items in a preferred genre. Users join at random times, so late joiners are cold at test time.
    """
    rng = np.random.default_rng(seed)
    movies, users, ratings = [], [], []
    item_genres = []
    for i in range(n_items):
        gs = rng.choice(len(GENRES), size=rng.integers(1, 4), replace=False)
        item_genres.append(set(gs.tolist()))
        year = int(rng.integers(1950, 2001))
        movies.append(f"{i + 1}::Movie {i + 1} ({year})::{'|'.join(GENRES[g] for g in gs)}")
    pop = rng.zipf(1.6, size=n_items).astype(float)
    pop = np.minimum(pop, 50.0)
    t0, span = 956_700_000, 90_000_000  # roughly the ML-1M time range (2000-04 .. 2003-02)
    for u in range(n_users):
        gender = rng.choice(["F", "M"])
        age = int(rng.choice(AGES))
        occ = int(rng.integers(0, N_OCC))
        users.append(f"{u + 1}::{gender}::{age}::{occ}::00000")
        base = (AGES.index(age) * 2 + (gender == "M")) % len(GENRES)
        prefs = {base, int(rng.integers(0, len(GENRES)))}
        match = np.array([len(item_genres[i] & prefs) > 0 for i in range(n_items)], dtype=float)
        p = pop * (1 + 3 * match)
        n = int(rng.integers(15, 50))
        items = rng.choice(n_items, size=min(n, n_items), replace=False, p=p / p.sum())
        active = int(rng.integers(1_000_000, 40_000_000))  # each user rates over days to months
        join = t0 + int(rng.random() * (span - active))
        times = join + np.sort(rng.integers(0, active, size=len(items)))
        for i, t in zip(items, times, strict=True):
            liked = match[i] > 0 and rng.random() < 0.85
            rating = int(rng.integers(4, 6)) if liked else int(rng.integers(1, 4))
            ratings.append(f"{u + 1}::{i + 1}::{rating}::{int(t)}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ml-1m/movies.dat", "\n".join(movies).encode("latin-1"))
        zf.writestr("ml-1m/users.dat", "\n".join(users).encode("latin-1"))
        zf.writestr("ml-1m/ratings.dat", "\n".join(ratings).encode("latin-1"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.getvalue())
