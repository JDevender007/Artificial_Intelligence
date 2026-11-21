from __future__ import annotations

from pathlib import Path

import pytest

from textclf.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["banking77.yaml", "smoke.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.models[-1] == "distilbert"


def test_overrides_are_yaml_parsed() -> None:
    cfg = load_config(None, ["seeds=[3, 4]", "bert.lr=1e-4", "tfidf.char_ngrams=[]", "data.max_train=50"])
    assert cfg.seeds == [3, 4] and cfg.bert.lr == pytest.approx(1e-4)
    assert cfg.tfidf.char_ngrams == [] and cfg.data.max_train == 50


@pytest.mark.parametrize("bad", ["models=[svm]", "seeds=[1, 1]", "data.val_fraction=0.7",
                                 "data.length_edges=[10, 5]"])
def test_validation_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        load_config(None, [bad])


def test_unknown_key_rejected() -> None:
    with pytest.raises(KeyError):
        load_config(None, ["bert.learning_rate=1"])
