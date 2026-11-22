"""One command per experiment.

    python scripts/run_experiment.py --config configs/squad.yaml
    python scripts/run_experiment.py --config configs/squad.yaml --set gen.variants=[rag] ft.modes=[rag]
"""

from __future__ import annotations

import argparse

from ragqa.config import load_config
from ragqa.experiment import run_experiment
from ragqa.utils import get_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    rows = run_experiment(cfg)
    get_logger().info("done: %d result rows in %s", len(rows), cfg.output_dir)


if __name__ == "__main__":
    main()
