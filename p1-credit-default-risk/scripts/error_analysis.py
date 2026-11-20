"""Error analysis + fairness audit of the final model on the test set.

Run after run_experiment.py with the same config:
    python scripts/error_analysis.py --config configs/default.yaml
Writes <output_dir>/error_analysis.md, slice CSVs and figures. All numbers are computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default import plots
from credit_default.config import load_config
from credit_default.error_analysis import add_slices, fairness_gaps, outcome_profile, slice_metrics
from credit_default.experiment import prepare_data
from credit_default.features import CreditFeatureEngineer

SLICES = ["recent_status", "months_delayed", "limit_quartile", "age_band", "sex", "education", "marriage"]
FAIRNESS_SLICES = ["sex", "age_band", "education", "marriage"]
PROFILE_COLS = ["LIMIT_BAL", "PAY_1", "max_delay", "n_months_delayed", "util_1", "util_mean",
                "payratio_1", "payratio_mean", "pay_total_to_limit"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg.output_dir)
    fig_dir = out / "figures"
    final = json.loads((out / "final_model.json").read_text())

    d = prepare_data(cfg)
    test = d["test"]
    y = d["y_te"].to_numpy()
    p = np.load(out / "preds" / f"seed{final['seed']}_{final['model']}.npy")
    pred = (p >= final["threshold"]).astype(int)

    plots.confusion(y, pred, fig_dir / "confusion_matrix.png")
    plots.score_histogram(y, p, final["threshold"], fig_dir / "score_hist.png")

    s = add_slices(test)
    md = [f"# Error analysis: `{final['model']}` (seed {final['seed']}, "
          f"threshold {final['threshold']:.3f})\n",
          "All numbers are computed on the held-out test set by `scripts/error_analysis.py`.\n"]

    tables = {}
    for col in SLICES:
        t = slice_metrics(y, p, pred, s[col])
        tables[col] = t
        t.to_csv(out / f"slice_{col}.csv", index=False)
        md += [f"\n## By `{col}`\n", t.to_markdown(index=False, floatfmt=".3f"), ""]

    md.append("\n## Fairness gaps (max − min across groups)\n")
    md.append("Protected attributes were " + ("**excluded** from" if cfg.features.drop_protected
              else "**included** in") + " the model's inputs; they are used here only to audit.\n")
    gaps = pd.DataFrame({c: fairness_gaps(tables[c]) for c in FAIRNESS_SLICES}).T
    gaps.to_csv(out / "fairness_gaps.csv")
    md += [gaps.to_markdown(floatfmt=".3f"), ""]

    feats = CreditFeatureEngineer(add_engineered=True, drop_protected=False).fit(d["X_te"]).transform(d["X_te"])
    prof = outcome_profile(feats, y, pred, PROFILE_COLS)
    prof.to_csv(out / "outcome_profile.csv")
    md += ["\n## What do missed defaulters (FN) look like vs caught ones (TP)?\n",
           prof.to_markdown(floatfmt=".3f"), ""]

    show = ["ID", "LIMIT_BAL", "AGE", *[f"PAY_{i}" for i in range(1, 4)], "BILL_AMT1", "PAY_AMT1"]
    fn = test.assign(p=p)[(y == 1) & (pred == 0)].nsmallest(10, "p")[show + ["p"]]
    fp = test.assign(p=p)[(y == 0) & (pred == 1)].nlargest(10, "p")[show + ["p"]]
    fn.to_csv(out / "worst_false_negatives.csv", index=False)
    fp.to_csv(out / "worst_false_positives.csv", index=False)
    md += ["\n## Most confident false negatives (defaulted, scored lowest)\n",
           fn.to_markdown(index=False, floatfmt=".3f"),
           "\n## Most confident false positives (did not default, scored highest)\n",
           fp.to_markdown(index=False, floatfmt=".3f"),
           "\n## Interpretation\n\n_Write 3-5 bullets here from the tables above (your own words)._\n"]

    (out / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
