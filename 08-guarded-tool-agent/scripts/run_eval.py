"""One command per experiment.

    python scripts/run_eval.py --config configs/eval.yaml
    python scripts/run_eval.py --config configs/smoke.yaml --set seeds=[0] agents=[agent_full]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from toolagent.config import load_config
from toolagent.evaluate import run_experiment
from toolagent.utils import get_logger
from toolagent.viewer import write_viewer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    ap.add_argument("--check", action="store_true", help="fail if the oracle with agent_full misses a task")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    run_experiment(cfg, check=a.check)
    n = write_viewer(Path(cfg.output_dir) / "traces", Path(cfg.output_dir) / "traces.html")
    get_logger().info("done: reports and %d traces in %s", n, cfg.output_dir)


if __name__ == "__main__":
    main()
