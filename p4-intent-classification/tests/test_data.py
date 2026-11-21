from __future__ import annotations

from pathlib import Path

import pytest

from textclf.config import DataConfig
from textclf.data import load_csv, make_splits, split_stats, stats_markdown


def test_load_csv_renames_and_cleans(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text('text,category\n"  my  card\nis late ",card_arrival\nhi,greeting\n', encoding="utf-8")
    df = load_csv(p)
    assert list(df.columns) == ["text", "intent"] and df.text[0] == "my card is late"


def test_load_csv_rejects_wrong_columns(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_csv(p)


def test_split_stratified_deterministic(splits, raw_frames) -> None:
    assert splits.labels == ["card_arrival", "exchange_rate", "lost_card"]
    assert len(splits.train) == 48 and len(splits.val) == 12 and len(splits.test) == 18
    assert splits.val.intent.value_counts().tolist() == [4, 4, 4]
    assert not set(splits.train.text) & set(splits.val.text)
    tr, te = (df.rename(columns={"category": "intent"}) for df in raw_frames)
    again = make_splits(tr, te, DataConfig(val_fraction=0.2, split_seed=0))
    assert again.val.text.tolist() == splits.val.text.tolist()


def test_max_train_subsamples(raw_frames) -> None:
    tr, te = (df.rename(columns={"category": "intent"}) for df in raw_frames)
    s = make_splits(tr, te, DataConfig(val_fraction=0.2, split_seed=0, max_train=12))
    assert len(s.train) == 12 and s.train.intent.nunique() == 3


def test_unseen_test_intent_rejected(raw_frames) -> None:
    tr, te = (df.rename(columns={"category": "intent"}) for df in raw_frames)
    te.loc[0, "intent"] = "new_intent"
    with pytest.raises(ValueError):
        make_splits(tr, te, DataConfig())


def test_stats(splits) -> None:
    st = split_stats(splits)
    assert st["n_classes"] == 3 and st["test"]["rows"] == 18 and st["test_exact_overlap_with_train"] == 0
    assert "| test | 18 | 3 | 6-6 |" in stats_markdown(st)
