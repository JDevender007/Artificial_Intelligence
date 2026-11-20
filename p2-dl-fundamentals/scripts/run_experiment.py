"""Run the full experiment from one command.

Examples:
    python scripts/run_experiment.py --config configs/cifar10.yaml
    python scripts/run_experiment.py --config configs/smoke.yaml
    python scripts/run_experiment.py --config configs/cifar10.yaml \
        --set output_dir=/kaggle/working/outputs/default train.epochs=5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dlfund.config import load_config
from dlfund.experiment import run_experiment
from dlfund.utils import get_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Override config values, e.g. train.epochs=5")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    logger = get_logger(log_file=Path(cfg.output_dir) / "run.log")
    logger.info("Config: %s", cfg.to_dict())
    run_experiment(cfg, logger)
    logger.info("Done. Results in %s/results.md", cfg.output_dir)


if __name__ == "__main__":
    main()
