from pathlib import Path

import pytest

from sdlora.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_load():
    main = load_config(ROOT / "configs/main.yaml")
    assert main.variants() == ["base", "lora_r8@400", "lora_r8@800"] and main.seeds == [0, 1, 2]
    smoke = load_config(ROOT / "configs/smoke.yaml")
    assert smoke.train.checkpoints[-1] == smoke.train.steps


def test_overrides_and_validation():
    c = load_config(None, ["train.rank=16", "train.steps=200", "train.checkpoints=[100,200]"])
    assert c.train.rank == 16 and c.variants()[-1] == "lora_r16@200"
    for bad in (["train.checkpoints=[400,300]"], ["data.resolution=500"], ["model.dtype=bf16"],
                ["safety.clip_threshold=1.5"], ["seeds=[0,0]"], ["nokeyvalue"]):
        with pytest.raises(ValueError):
            load_config(None, bad)
