"""Regenerate ``notebooks/kaggle_run.ipynb``. The notebook is a thin wrapper; all logic is in ``src/rlinv``.

    python scripts/make_notebook.py [--configs configs/smoke.yaml configs/main.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P10 RL (CartPole + custom inventory env): Kaggle runner

Thin wrapper: all logic lives in `src/rlinv`.

Settings: **CPU (no accelerator), Internet ON**, then *Save Version -> Save & Run All*. Jobs run in a process pool
with one worker per CPU. The smoke config runs first with `--check`, so the notebook stops if any run or plot is
missing. Download `p10_results.zip` from the Output tab: results, bootstrap, learning curves (png), runs.json
(every curve and every held-out episode) and the config. Nothing here is downloaded except pip packages: CartPole
ships with gymnasium and the inventory env is generated."""

VERSIONS = (
    "import os, torch, gymnasium, stable_baselines3, numpy, scipy, matplotlib, yaml; "
    "[print(m.__name__, m.__version__) for m in (torch, gymnasium, stable_baselines3, numpy, scipy, matplotlib, yaml)]"
    "; print('cpus', os.cpu_count())"
)


def build(configs: list[str]) -> dict:
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/p10-rl-inventory\n"
                 "!pip install -q gymnasium==1.3.0 stable-baselines3==2.9.0 && pip install -q -e . --no-deps"),
        ("code", "!mkdir -p /kaggle/working/outputs\n"
                 f"!python -c \"{VERSIONS}\" | tee /kaggle/working/outputs/env.txt\n"
                 "!lscpu | grep -E 'Model name|^CPU\\(s\\)' | tee -a /kaggle/working/outputs/env.txt\n"
                 "!pip freeze | grep -iE '^(torch|gymnasium|stable.baselines3|numpy|scipy|matplotlib|pyyaml)==' "
                 "| tee /kaggle/working/outputs/versions.txt"),
    ]
    for config in configs:
        name = Path(config).stem
        out = f"/kaggle/working/outputs/{name}"
        cells += [
            ("markdown", f"## Run `{config}`"),
            ("code", "import subprocess\nimport sys\n\n"
                     f"subprocess.run([sys.executable, 'scripts/run_experiment.py', '--config', '{config}', '--check', "
                     f"'--set', 'output_dir={out}'], check=True)"),
        ]
    cells += [
        ("markdown", "## Package results"),
        ("code", "%cd /kaggle/working\n"
                 "!zip -qr p10_results.zip outputs\n"
                 "!rm -rf outputs Artificial_Intelligence && ls -la /kaggle/working"),
    ]
    nb_cells = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb_cells.append(c)
    return {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                           "name": "python3"},
                                            "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["configs/smoke.yaml", "configs/main.yaml"])
    ap.add_argument("--out", default="notebooks/kaggle_run.ipynb")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(build(a.configs), indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
