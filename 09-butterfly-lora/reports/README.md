# reports/ — not populated yet

This folder is where the committed snapshot of the first completed run goes. It is empty on purpose:
no image in this project has been generated and scored yet, and inventing a number would break the
rule the whole portfolio is built on.

A completed run writes to `outputs/main/` (gitignored). Copy the reports in:

```powershell
python scripts\run_experiment.py --config configs\main.yaml --check
robocopy outputs\main reports\kaggle_t4 /E /XF *.safetensors *.bin
```

| file | what it contains |
|---|---|
| `results.md` | mean ± std over seeds for base and both LoRA checkpoints: FID, KID×1000, CLIP score, top-1/top-5, memorisation, clip-flag rate, ms/image, plus the real-vs-real floor row |
| `bootstrap.md` | paired bootstrap of each variant against `base` |
| `safety.md` | prompt-filter categories, image-filter false-positive rates on benign prompts, provenance |
| `training.md` | the LoRA training log — loss, lr, steps, seconds |
| `grids/`, `gen/` | contact sheets and every generated PNG with provenance metadata |

Once `reports/kaggle_t4/results.md` exists, this file is deleted and the README results table is filled
from it — copied, never retyped.
