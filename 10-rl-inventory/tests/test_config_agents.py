from pathlib import Path

import pytest

from rlinv.agents import parse_schedule, train_one
from rlinv.config import load_config
from rlinv.experiment import build_jobs

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs():
    main = load_config(ROOT / "configs/main.yaml")
    assert main.seeds == [0, 1, 2, 3, 4] and [e.id for e in main.envs] == ["CartPole-v1", "Inventory-v0"]
    jobs = build_jobs(main)
    assert len(jobs) == 2 * 5 * 5 and sum(j["keep_model"] for j in jobs) == 2
    smoke = load_config(ROOT / "configs/smoke.yaml")
    assert max(j["total_steps"] for j in build_jobs(smoke)) == 6000


def test_overrides_and_validation():
    c = load_config(ROOT / "configs/smoke.yaml", ["seeds=[3]", "n_workers=1"])
    assert c.seeds == [3] and c.n_workers == 1
    for bad in (["seeds=[0,0]"], ["steps_scale=0"], ["final_seed_base=50001"], ["nokey"], ["bogus=1"],
                ["envs=[{id: CartPole-v1, algos: [a2c]}]"]):
        with pytest.raises(ValueError):
            load_config(ROOT / "configs/smoke.yaml", bad)


def test_parse_schedule():
    f = parse_schedule("lin_0.001")
    assert f(1.0) == 0.001 and f(0.5) == 0.0005 and parse_schedule(3e-4) == 3e-4


@pytest.mark.parametrize("algo", ["dqn_notarget", "sb3_ppo"])
def test_train_one_tiny(algo):
    hp = {"learning_starts": 50, "buffer_size": 1000} if algo.startswith("dqn") else {"n_envs": 2, "n_steps": 64,
                                                                                         "batch_size": 64}
    out = train_one({"env_id": "Inventory-v0", "algo": algo, "seed": 0, "total_steps": 256, "eval_every": 128,
                     "eval_seeds": [1], "final_seeds": [2, 3], "hp": hp, "env_kwargs": {"horizon": 10}})
    assert [s for s, _ in out["curve"]] == [0, 128, 256] and len(out["final"]) == 2
    assert all(ep["length"] == 10 for ep in out["final"])
