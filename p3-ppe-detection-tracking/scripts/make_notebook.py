"""Regenerate ``notebooks/kaggle_run.ipynb`` (thin wrapper; all logic is in ``src/ppedet``).

    python scripts/make_notebook.py [--config configs/smoke.yaml --out other.ipynb]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P3 PPE detection + tracking: Kaggle runner

Thin wrapper: all logic lives in `src/ppedet`.

Settings: **Accelerator GPU T4 x1**, **Internet ON**, then *Save Version -> Save & Run All*.
The run is resumable: rerunning skips finished (variant, seed) rows and resumes from `last.pt`.

Download `p3_results.zip` from the Output tab at the end (metrics, plots, error analysis,
tracking tables and the main model's `best.pt`)."""


def build(config: str) -> dict:
    name = Path(config).stem
    out = f"/kaggle/working/outputs/{name}"
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/p3-ppe-detection-tracking\n"
                 "!pip install -q ultralytics==8.4.216 tabulate && pip install -q -e . --no-deps\n"
                 "!yolo settings datasets_dir=/kaggle/working/datasets sync=False"),
        ("code", f"!mkdir -p {out}\n"
                 f"!python -c \"import ultralytics, torch, numpy, cv2; print('ultralytics', ultralytics.__version__); "
                 f"print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('numpy', numpy.__version__); "
                 f"print('opencv', cv2.__version__)\" | tee {out}/env.txt\n"
                 f"!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a {out}/env.txt"),
        ("markdown", f"## Run `{config}`\nSplit + data stats, every variant x seed (train, then test once), "
                     "error analysis + compliance on the main model, then ByteTrack vs BoT-SORT."),
        ("code", f"!python scripts/run_experiment.py --config {config} --set output_dir={out}"),
        ("code", "from pathlib import Path\n\n"
                 f"for f in ['data_summary.md', 'results.md', 'per_class.md', 'error_analysis.md', 'tracking.md']:\n"
                 f"    p = Path('{out}') / f\n"
                 "    print(f'===== {f}')\n"
                 "    print(p.read_text() if p.exists() else '(missing)')"),
        ("markdown", "## Package results\nThe zip holds text results for every run plus the main run's key plots, "
                     "kept small on purpose. The main model's weights stay as a separate output file. "
                     "Everything else (dataset, clone, other weights) is deleted so the kernel output stays small."),
        ("code", "%cd /kaggle/working\n"
                 f"!cp outputs/{name}/runs/pretrained_n_s0/weights/best.pt best_pretrained_n_s0.pt || true\n"
                 f"!mkdir -p outputs/{name}/figures && for f in results.png confusion_matrix_normalized.png "
                 f"BoxPR_curve.png; do cp outputs/{name}/runs/pretrained_n_s0/$f outputs/{name}/figures/ 2>/dev/null; "
                 f"done; cp outputs/{name}/test/pretrained_n_s0/confusion_matrix_normalized.png "
                 f"outputs/{name}/figures/test_confusion_matrix_normalized.png 2>/dev/null || true\n"
                 "!zip -qr p3_results.zip outputs -x '*/weights/*' '*/split/*.txt' '*/runs/*.jpg' '*/runs/*.png' "
                 "'*/test/*.jpg' '*/test/*.png'\n"
                 "!rm -rf outputs datasets Artificial_Intelligence *.pt.tmp && ls -la /kaggle/working"),
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
    ap.add_argument("--config", default="configs/ppe.yaml")
    ap.add_argument("--out", default="notebooks/kaggle_run.ipynb")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(build(a.config), indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
