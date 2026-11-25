"""Regenerate ``notebooks/kaggle_run.ipynb`` (thin wrapper; all logic is in ``src/toolagent``).

    python scripts/make_notebook.py [--configs configs/smoke.yaml configs/eval.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P8 tool-using agent: Kaggle runner

Thin wrapper: all logic lives in `src/toolagent`.

Settings: **GPU T4, Internet ON**, then *Save Version -> Save & Run All*. Runs are resumable (a finished
(model, agent, seed) is skipped). The smoke config runs with `--check`, so the notebook stops if the oracle fails a
task or the real model cannot load.

Download `p8_results.zip` from the Output tab: results, per-category table, bootstrap, failure taxonomy, every trace
(`traces/*.jsonl`) and the static trace viewer (`traces.html`). Everything is synthetic (no dataset licence)."""

VERSIONS = (
    "import torch, transformers, bitsandbytes, accelerate, numpy, pandas, yaml; "
    "[print(m.__name__, m.__version__) for m in (torch, transformers, bitsandbytes, accelerate, numpy, pandas, yaml)]"
    "; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
)


def build(configs: list[str]) -> dict:
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/08-tool-agent\n"
                 "!pip install -q -r requirements-gpu.txt && pip install -q -e . --no-deps"),
        ("code", "!mkdir -p /kaggle/working/outputs\n"
                 f"!python -c \"{VERSIONS}\" | tee /kaggle/working/outputs/env.txt\n"
                 "!(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo no-gpu) "
                 "| tee -a /kaggle/working/outputs/env.txt\n"
                 "!pip freeze | grep -iE '^(torch|transformers|bitsandbytes|accelerate|tokenizers|safetensors|"
                 "huggingface.hub|numpy|pandas|pyyaml)==' | tee /kaggle/working/outputs/versions.txt"),
    ]
    for config in configs:
        name = Path(config).stem
        out = f"/kaggle/working/outputs/{name}"
        cells += [
            ("markdown", f"## Run `{config}`"),
            ("code", "import subprocess\nimport sys\n\n"
                     f"subprocess.run([sys.executable, 'scripts/run_eval.py', '--config', '{config}', '--check', "
                     f"'--set', 'output_dir={out}'], check=True)"),
            ("code", "from pathlib import Path\n\n"
                     "for f in ['results.md', 'categories.md', 'bootstrap.md', 'failures.md']:\n"
                     f"    p = Path('{out}') / f\n"
                     "    print(f'===== {f}')\n"
                     "    print(p.read_text() if p.exists() else '(missing)')"),
        ]
    cells += [
        ("markdown", "## Package results"),
        ("code", "%cd /kaggle/working\n"
                 "!zip -qr p8_results.zip outputs -x '*/world/*'\n"
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
    ap.add_argument("--configs", nargs="+", default=["configs/smoke.yaml", "configs/eval.yaml"])
    ap.add_argument("--out", default="notebooks/kaggle_run.ipynb")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(build(a.configs), indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
