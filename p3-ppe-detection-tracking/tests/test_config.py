from __future__ import annotations

from pathlib import Path

import pytest

from ppedet.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["ppe.yaml", "smoke.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.eval.main_variant in [v.name for v in cfg.variants]


def test_full_config_has_baseline_and_ablations() -> None:
    cfg = load_config(ROOT / "configs" / "ppe.yaml")
    by = {v.name: v for v in cfg.variants}
    assert by["scratch_n"].model.endswith(".yaml") and not by["scratch_n"].pretrained
    assert by["pretrained_n_frozen"].overrides["freeze"] == 10
    assert len(cfg.seeds) >= 3


def test_overrides_parse_yaml_values() -> None:
    cfg = load_config(ROOT / "configs" / "smoke.yaml", ["train.epochs=3", "seeds=[1, 2]"])
    assert cfg.train.epochs == 3 and cfg.seeds == [1, 2]


def test_unknown_key_rejected() -> None:
    with pytest.raises(KeyError):
        load_config(None, ["train.epoch=3"])


def test_yaml_model_must_not_be_pretrained() -> None:
    with pytest.raises(ValueError):
        load_config(None, ["variants=[{name: pretrained_n, model: yolo26n.yaml}]"])
