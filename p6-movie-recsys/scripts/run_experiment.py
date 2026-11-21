"""One command per experiment.

    python scripts/run_experiment.py --config configs/ml1m.yaml
    python scripts/run_experiment.py --config configs/ml1m.yaml --set "models=[popularity,bpr_mf]"
"""

from __future__ import annotations

import argparse

from movierec.config import load_config
from movierec.experiment import run_experiment
from movierec.utils import get_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    out = run_experiment(cfg)
    get_logger().info("done: %s", out)


if __name__ == "__main__":
    main()
