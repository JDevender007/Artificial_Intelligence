from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from movierec.config import load_config, tower_variant
from movierec.data import GENRES, _year_bucket, summary
from movierec.models import causal_histories

ROOT = Path(__file__).resolve().parents[1]


def test_overrides_and_validation():
    cfg = load_config(ROOT / "configs" / "ml1m.yaml", ["tower.temperature=1e-1", "seeds=[3,4,5]"])
    assert cfg.tower.temperature == pytest.approx(0.1) and cfg.seeds == [3, 4, 5]
    assert cfg.bpr.lr == pytest.approx(5e-3)
    with pytest.raises(ValueError):
        load_config(None, ["models=[nope]"])
    with pytest.raises(ValueError):
        load_config(None, ["data.val_frac=0.6", "data.test_frac=0.5"])
    with pytest.raises(KeyError):
        load_config(None, ["tower.nope=1"])
    assert not tower_variant(cfg.tower, "two_tower_nofeat").use_features
    assert tower_variant(cfg.tower, "two_tower_inbatch").n_random_neg == 0


def test_year_bucket():
    assert _year_bucket("Toy Story (1995)") == 8
    assert _year_bucket("Old (1902)") == 0
    assert _year_bucket("No year") == 10


def test_temporal_split_has_no_overlap(ds):
    r = ds.ratings
    assert r.loc[r.split == "train", "ts"].max() < ds.t_val <= r.loc[r.split == "val", "ts"].min()
    assert r.loc[r.split == "val", "ts"].max() < ds.t_test <= r.loc[r.split == "test", "ts"].min()
    assert set(r.split) == {"train", "val", "test"}
    s = summary(ds)
    assert s["train_ratings"] + s["val_ratings"] + s["test_ratings"] == s["ratings"]
    assert s["test_cold_users"] > 0  # late joiners in the synthetic data


def test_features_and_seen(ds):
    assert ds.user_feat.shape == (ds.n_users, 3)
    assert ds.item_genres.max() <= len(GENRES) and ds.item_genres.min() >= 0
    seen = ds.seen_matrix("train", "val")
    assert seen.nnz == len(ds.part("train", "val"))


def test_history_uses_only_cutoff_splits(ds):
    users = np.arange(ds.n_users)
    h = ds.history(users, "train", length=5)
    train_pos = ds.part("train", positives=True)
    for u in users[:50]:
        got = h[u][h[u] >= 0]
        want = train_pos[train_pos.u == u]["i"].to_numpy()[-5:]
        assert np.array_equal(got, want)


def test_causal_histories_exclude_current_and_future(ds):
    u, i, h = causal_histories(ds, length=4)
    p = ds.part("train", positives=True)
    assert len(u) == len(p)
    for row in range(0, len(u), 37):
        items = p[p.u == u[row]]["i"].to_numpy()
        pos = np.flatnonzero(items == i[row])[0]
        prev = h[row][h[row] >= 0]
        assert np.array_equal(prev, items[max(0, pos - 4):pos])
