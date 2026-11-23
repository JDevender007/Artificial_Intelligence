"""Regenerate ``notebooks/kaggle_run.ipynb`` (thin wrapper; all logic is in ``src/ragqa``).

    python scripts/make_notebook.py [--configs configs/smoke.yaml configs/squad.yaml --out other.ipynb]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P7 RAG + QLoRA on SQuAD 2.0: Kaggle runner

Thin wrapper: all logic lives in `src/ragqa`.

Settings: **GPU T4, Internet ON**, then *Save Version -> Save & Run All*. Every stage is resumable: rerunning skips
retrieval rows, prediction files and adapters that already exist. Each config runs with `check=True`, so a failing
smoke run stops the notebook before the long run.

Download `p7_results.zip` from the Output tab: retrieval ablation, predictions, results, bootstrap, error analysis,
fine-tuning logs and the environment. LoRA adapters (~70 MB each) are left out. SQuAD 2.0 is CC BY-SA 4.0: files that
quote passages or questions are share-alike."""

VERSIONS = (
    "import torch, transformers, peft, bitsandbytes, accelerate, numpy, pandas, scipy, yaml; "
    "[print(m.__name__, m.__version__) for m in (torch, transformers, peft, bitsandbytes, accelerate, numpy, "
    "pandas, scipy, yaml)]; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
)


def build(configs: list[str]) -> dict:
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/07-rag-qlora\n"
                 "!pip install -q peft bitsandbytes accelerate tabulate && pip install -q -e . --no-deps\n"
                 "# Kaggle ships torchao 0.10; peft >= 0.18 refuses to load adapters next to torchao < 0.16. Not used here.\n"
                 "!pip uninstall -q -y torchao"),
        ("code", "!mkdir -p /kaggle/working/outputs\n"
                 f"!python -c \"{VERSIONS}\" | tee /kaggle/working/outputs/env.txt\n"
                 "!(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo no-gpu) "
                 "| tee -a /kaggle/working/outputs/env.txt\n"
                 "!pip freeze | grep -iE '^(torch|transformers|peft|bitsandbytes|accelerate|tokenizers|safetensors|"
                 "huggingface.hub|numpy|pandas|scipy|pyyaml|tabulate)==' | tee /kaggle/working/outputs/versions.txt"),
    ]
    for config in configs:
        name = Path(config).stem
        out = f"/kaggle/working/outputs/{name}"
        cells += [
            ("markdown", f"## Run `{config}`"),
            ("code", "import subprocess\nimport sys\n\n"
                     f"subprocess.run([sys.executable, 'scripts/run_experiment.py', '--config', '{config}', "
                     f"'--set', 'output_dir={out}', 'data.raw_dir=/kaggle/working/data'], check=True)"),
            ("code", "from pathlib import Path\n\n"
                     "for f in ['data_summary.json', 'retrieval.md', 'retrieval_choice.json', 'results.md', "
                     "'bootstrap.md', 'error_analysis.md']:\n"
                     f"    p = Path('{out}') / f\n"
                     "    print(f'===== {f}')\n"
                     "    print(p.read_text() if p.exists() else '(missing)')"),
        ]
    cells += [
        ("markdown", "## Package results (adapters and raw data left out)"),
        ("code", "%cd /kaggle/working\n"
                 "!zip -qr p7_results.zip outputs -x '*/adapters/*'\n"
                 "!rm -rf outputs data Artificial_Intelligence && ls -la /kaggle/working"),
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
    ap.add_argument("--configs", nargs="+", default=["configs/smoke.yaml", "configs/squad.yaml"])
    ap.add_argument("--out", default="notebooks/kaggle_run.ipynb")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(build(a.configs), indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
