"""Run one P3 experiment from a YAML config.

    python scripts/run_experiment.py --config configs/ppe.yaml [--set train.epochs=10 "seeds=[0]"]
"""

from __future__ import annotations

import argparse
import json

from ppedet.config import load_config
from ppedet.experiment import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=5")
    a = ap.parse_args()
    summary = run_experiment(load_config(a.config, a.set))
    print(json.dumps(summary.get("results", []), indent=2, default=float))


if __name__ == "__main__":
    main()
