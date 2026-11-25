# reports/ — not populated yet

This folder is where the committed snapshot of the first completed run goes. It is empty on purpose:
no number in this project has been measured yet, and inventing one would break the rule the whole
portfolio is built on.

A completed run writes to `outputs/squad/` (gitignored). Copy the reports in:

```powershell
python scripts\run_experiment.py --config configs\squad.yaml
robocopy outputs\squad reports\kaggle_t4 /E /XF *.safetensors *.bin
```

| file | what it contains |
|---|---|
| `results.md` | mean ± std over seeds: EM, F1, answerable F1, no-answer accuracy, false abstain, context recall, faithfulness, citation rate, ms/question |
| `retrieval.md` | the ablation — val and test recall@{1,5,20}, MRR@10, ms/query, 2 chunkers × 4 methods |
| `bootstrap.md` | per-question F1 with a 95% bootstrap CI and the paired CI of the difference vs `rag` |
| `error_analysis.md` | outcome buckets and failure examples |
| `predictions/*.jsonl` | every answer with its retrieved passages and cited id |

Once `reports/kaggle_t4/results.md` exists, this file is deleted and the README results table is filled
from it — copied, never retyped.
