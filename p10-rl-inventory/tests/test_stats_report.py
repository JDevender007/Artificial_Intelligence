import json

import numpy as np

from rlinv.report import write_reports
from rlinv.stats import hierarchical_bootstrap, iqm, t_interval


def test_t_interval():
    m, lo, hi = t_interval(np.array([[1.0, 2.0], [3.0, 2.0]]))
    assert m.tolist() == [2.0, 2.0] and lo[1] == hi[1] == 2.0 and lo[0] < 2 < hi[0]
    m, lo, hi = t_interval(np.array([[5.0]]))
    assert m[0] == lo[0] == hi[0] == 5.0


def test_bootstrap_and_iqm():
    rng = np.random.default_rng(0)
    eps = rng.normal(size=40)
    a = eps[None, :] + 1.0 + rng.normal(scale=0.01, size=(5, 40))
    r = hierarchical_bootstrap(a, eps[None, :], n=500)
    assert abs(r["diff"] - 1.0) < 0.05 and r["ci_low"] > 0.9 and r["p_le_0"] == 0.0
    assert iqm([0, 1, 2, 3, 100, -100, 1, 2]) == 1.5


def _run(algo, seed, base, curve=True, heuristic=False):
    final = [{"seed": 100 + i, "score": base + seed + i, "length": 10} for i in range(4)]
    return {"env_id": "CartPole-v1", "algo": algo, "seed": seed, "final": final, "train_s": 1.0,
            "curve": [(0, 10.0), (100, base + seed)] if curve else [], **({"heuristic": True, "params": {"S": 1}}
                                                                         if heuristic else {})}


def test_write_reports(tmp_path):
    runs = [_run(a, s, b) for a, b in (("dqn", 480), ("sb3_ppo", 300)) for s in (0, 1, 2)]
    runs.append(_run("sS", 0, 200, curve=False, heuristic=True))
    res = write_reports(tmp_path, runs, {"CartPole-v1": 475.0}, n_bootstrap=100)
    s = res["CartPole-v1"]["summary"]
    assert s["dqn"]["solved_seeds"] == 3 and s["dqn"]["median_steps_to_threshold"] == 100
    assert s["sb3_ppo"]["solved_seeds"] == 0 and s["sS"]["n_seeds"] == 1
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert md.index("| sS") < md.index("| dqn") < md.index("| sb3_ppo")
    boot = (tmp_path / "bootstrap.md").read_text(encoding="utf-8")
    assert "sb3_ppo - dqn" in boot and "dqn - sS" in boot
    assert (tmp_path / "curves_CartPole-v1.png").exists()
    assert len(json.loads((tmp_path / "runs.json").read_text())) == 7
