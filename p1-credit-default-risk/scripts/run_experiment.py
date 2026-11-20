"""Run the full experiment from one command.

Examples:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/smoke.yaml
    python scripts/run_experiment.py --config configs/default.yaml \
        --set output_dir=/kaggle/working/outputs/default tuning.n_trials=30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from credit_default.config import load_config
from credit_default.experiment import run_experiment
from credit_default.utils import get_logger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Override config values, e.g. tuning.n_trials=10")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    logger = get_logger(log_file=Path(cfg.output_dir) / "run.log")
    logger.info("Config: %s", cfg.to_dict())
    run_experiment(cfg, logger)
    logger.info("Done. Results in %s/results.md", cfg.output_dir)


if __name__ == "__main__":
    main()
