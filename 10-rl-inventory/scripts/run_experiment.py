"""One command per experiment.

    python scripts/run_experiment.py --config configs/smoke.yaml --check     # minutes on CPU
    python scripts/run_experiment.py --config configs/main.yaml --check      # Kaggle CPU, 4 workers
    python scripts/run_experiment.py --config configs/main.yaml --set n_workers=2 seeds=[0,1,2]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rlinv.config import load_config
from rlinv.experiment import run_experiment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. seeds=[0,1] n_workers=2")
    ap.add_argument("--check", action="store_true", help="fail if any expected report or run is missing")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config, a.set)
    res = run_experiment(cfg)
    out = Path(cfg.output_dir)
    print((out / "results.md").read_text(encoding="utf-8"))
    print((out / "bootstrap.md").read_text(encoding="utf-8"))
    if a.check:
        problems = []
        for e in cfg.envs:
            summ = res.get(e.id, {}).get("summary", {})
            for name in [*e.algos, *e.heuristics]:
                if name not in summ:
                    problems.append(f"{e.id}: missing {name}")
                elif name in e.algos and summ[name]["n_seeds"] != len(cfg.seeds):
                    problems.append(f"{e.id}: {name} has {summ[name]['n_seeds']} seeds")
            if not (out / f"curves_{e.id}.png").exists():
                problems.append(f"missing curves_{e.id}.png")
        if problems:
            print("CHECK FAILED:\n  " + "\n  ".join(problems))
            return 1
        print("CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
