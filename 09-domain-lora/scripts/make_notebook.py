"""Regenerate ``notebooks/kaggle_run.ipynb`` (thin wrapper; all logic is in ``src/sdlora``).

    python scripts/make_notebook.py [--configs configs/smoke.yaml configs/main.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INTRO = """# P9 SD 1.5 butterfly LoRA: Kaggle runner

Thin wrapper: all logic lives in `src/sdlora`.

Settings: **GPU T4, Internet ON**, then *Save Version -> Save & Run All*. Runs are resumable (a finished
(variant, seed) is skipped; trained LoRAs are reused). The smoke config runs with `--check`, so the notebook stops if
a run is missing, a metric is not finite or the LoRA does not change the images.

Download `p9_results.zip` from the Output tab: results, bootstrap, safety and training tables, per-prompt metrics,
training loss logs and small sample grids. Generated images, the dataset copy, feature caches and LoRA weights stay
out of the zip (size). Data: Smithsonian butterflies, CC0-1.0. Model: SD 1.5, CreativeML OpenRAIL-M."""

VERSIONS = (
    "import torch, transformers, diffusers, peft, accelerate, torchmetrics, numpy, pandas, PIL, yaml; "
    "[print(m.__name__, m.__version__) for m in (torch, transformers, diffusers, peft, accelerate, torchmetrics, "
    "numpy, pandas, PIL, yaml)]"
    "; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
)


def build(configs: list[str]) -> dict:
    cells = [
        ("markdown", INTRO),
        ("code", "!git clone --depth 1 https://github.com/JDevender007/Artificial_Intelligence.git\n"
                 "%cd Artificial_Intelligence/09-domain-lora\n"
                 "!pip install -q -r requirements-gpu.txt && pip install -q -e . --no-deps\n"
                 "!pip uninstall -y -q torchao  # peft refuses to build LoRA layers next to Kaggle's torchao 0.10"),
        ("code", "!mkdir -p /kaggle/working/outputs\n"
                 f"!python -c \"{VERSIONS}\" | tee /kaggle/working/outputs/env.txt\n"
                 "!(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo no-gpu) "
                 "| tee -a /kaggle/working/outputs/env.txt\n"
                 "!pip freeze | grep -iE '^(torch|transformers|accelerate|tokenizers|safetensors|"
                 "huggingface.hub|diffusers|peft|torchmetrics|torch.fidelity|numpy|pandas|pillow|pyyaml)==' "
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
            ("code", "from pathlib import Path\n\n"
                     "for f in ['results.md', 'bootstrap.md', 'safety.md', 'training.md']:\n"
                     f"    p = Path('{out}') / f\n"
                     "    print(f'===== {f}')\n"
                     "    print(p.read_text() if p.exists() else '(missing)')"),
        ]
    cells += [
        ("markdown", "## Package results"),
        ("code", "%cd /kaggle/working\n"
                 "!zip -qr p9_results.zip outputs -x '*/data/*' '*/gen/*' '*/features/*' '*.safetensors'\n"
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
