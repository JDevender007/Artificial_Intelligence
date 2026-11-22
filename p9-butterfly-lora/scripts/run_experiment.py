"""Run one experiment end to end (GPU required).

    python scripts/run_experiment.py --config configs/main.yaml [--set train.steps=400] [--check]
"""

from __future__ import annotations

import argparse

from sdlora.config import load_config
from sdlora.experiment import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides such as train.rank=16")
    ap.add_argument("--check", action="store_true", help="assert every run finished and the LoRA changes images")
    a = ap.parse_args()
    run_experiment(load_config(a.config, a.set), check=a.check)


if __name__ == "__main__":
    main()
