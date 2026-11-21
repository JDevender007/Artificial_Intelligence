"""Regenerate ``notebooks/kaggle_run.ipynb`` (thin wrapper; all logic is in ``src/textclf``).

    python scripts/make_notebook.py [--config configs/smoke.yaml --out other.ipynb]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P4 Banking77 intent classification: Kaggle runner

Thin wrapper: all logic lives in `src/textclf`.

Settings: **Accelerator GPU T4 x1**, **Internet ON**, then *Save Version -> Save & Run All*.
The run can be resumed: rerunning skips (model, seed) rows that already finished.

Download `p4_results.zip` from the Output tab at the end. It holds the metrics, per-class results,
error analysis, latency numbers, the model card and the test predictions. Model weights are left out."""


def build(config: str) -> dict:
    name = Path(config).stem
    out = f"/kaggle/working/outputs/{name}"
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/p4-intent-classification\n"
                 "!pip install -q tabulate && pip install -q -e . --no-deps"),
        ("code", f"!mkdir -p {out}\n"
                 "!python -c \"import transformers, tokenizers, torch, sklearn, numpy; "
                 "print('transformers', transformers.__version__); print('tokenizers', tokenizers.__version__); "
                 "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
                 f"print('scikit-learn', sklearn.__version__); print('numpy', numpy.__version__)\" | tee {out}/env.txt\n"
                 f"!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a {out}/env.txt\n"
                 f"!lscpu | grep 'Model name' | tee -a {out}/env.txt"),
        ("markdown", f"## Run `{config}`\nDownloads Banking77, makes the split, fits every model x seed, measures "
                     "latency, then writes the error analysis and model card."),
        ("code", f"!python scripts/run_experiment.py --config {config} --set output_dir={out}"),
        ("code", "from pathlib import Path\n\n"
                 "for f in ['data_summary.md', 'results.md', 'latency.md', 'error_analysis.md']:\n"
                 f"    p = Path('{out}') / f\n"
                 "    print(f'===== {f}')\n"
                 "    print(p.read_text() if p.exists() else '(missing)')"),
        ("markdown", "## Package results\nThe zip holds text results and predictions only (no weights)."),
        ("code", "%cd /kaggle/working\n"
                 "!zip -qr p4_results.zip outputs -x '*/models/*'\n"
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
    ap.add_argument("--config", default="configs/banking77.yaml")
    ap.add_argument("--out", default="notebooks/kaggle_run.ipynb")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(build(a.config), indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
