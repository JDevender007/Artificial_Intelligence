"""Regenerate ``notebooks/kaggle_run.ipynb`` (thin wrapper; all logic is in ``src/movierec``).

    python scripts/make_notebook.py [--config configs/smoke.yaml --out other.ipynb]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P6 MovieLens-1M recommender: Kaggle runner

Thin wrapper: all logic lives in `src/movierec`.

Settings: **GPU T4, Internet ON**, then *Save Version -> Save & Run All*. The run can be resumed: rerunning skips
(model, seed) rows that already finished.

Download `p6_results.zip` from the Output tab. It holds the metrics, segment and bootstrap tables, the FAISS
benchmark, the error analysis and `serve/` (seed-0 two-tower checkpoint + item vectors for the API). Per-user
arrays and the raw data are left out. MovieLens may not be redistributed: keep `serve/meta.json` out of git."""

VERSIONS = (
    "import torch, numpy, pandas, scipy, faiss, fastapi; "
    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
    "print('numpy', numpy.__version__); print('pandas', pandas.__version__); "
    "print('scipy', scipy.__version__); print('faiss', faiss.__version__); print('fastapi', fastapi.__version__)"
)


def build(config: str) -> dict:
    name = Path(config).stem
    out = f"/kaggle/working/outputs/{name}"
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/06-two-tower-recsys\n"
                 "!pip install -q -r requirements-gpu.txt && pip install -q -e . --no-deps"),
        ("code", f"!mkdir -p {out}\n"
                 f"!python -c \"{VERSIONS}\" | tee {out}/env.txt\n"
                 f"!(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo no-gpu) | tee -a {out}/env.txt\n"
                 f"!lscpu | grep 'Model name' | tee -a {out}/env.txt"),
        ("markdown", f"## Run `{config}`\nDownloads ml-1m.zip from GroupLens, splits by time, trains every model x "
                     "seed, then writes the reports and the FAISS benchmark."),
        ("code", f"!python scripts/run_experiment.py --config {config} --set output_dir={out}"),
        ("code", "from pathlib import Path\n\n"
                 "for f in ['data_summary.md', 'results.md', 'segments.md', 'bootstrap.md', 'index.md', "
                 "'error_analysis.md']:\n"
                 f"    p = Path('{out}') / f\n"
                 "    print(f'===== {f}')\n"
                 "    print(p.read_text() if p.exists() else '(missing)')"),
        ("markdown", "## Package results\nThe zip holds text results and the serving artefacts (no per-user arrays)."),
        ("code", "%cd /kaggle/working\n"
                 "!zip -qr p6_results.zip outputs -x '*/per_user/*'\n"
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
    ap.add_argument("--config", default="configs/ml1m.yaml")
    ap.add_argument("--out", default="notebooks/kaggle_run.ipynb")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(build(a.config), indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
