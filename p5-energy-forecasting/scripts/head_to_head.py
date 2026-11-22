"""Paired day-level bootstrap between the top models, plus a 3-seed average ensemble, from saved forecasts.

    python scripts/head_to_head.py --forecasts outputs/household_power/forecasts --out outputs/household_power/head_to_head.md

Run after ``run_experiment.py``. This reads only the forecasts already written by the backtest; nothing is
re-trained or tuned on the test period. The ensemble is the plain mean of the seeds' forecasts (no weights).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsforecast.metrics import paired_bootstrap


def _load(d: Path, model: str, seed: int) -> pd.DataFrame:
    return pd.read_parquet(d / f"{model}_s{seed}.parquet")


def seed_average(d: Path, model: str, seeds: list[int]) -> pd.DataFrame:
    """Forecast frame whose ``yhat`` is the mean over seeds (all seeds share origins and steps)."""
    frames = [_load(d, model, s) for s in seeds]
    out = frames[0].copy()
    for f in frames[1:]:
        if not (f.ts.to_numpy() == out.ts.to_numpy()).all():
            raise ValueError(f"{model}: seeds have different forecast timestamps")
    out["yhat"] = sum(f.yhat.to_numpy() for f in frames) / len(frames)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasts", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    d = a.forecasts
    rows = []
    pairs = [("sarima", "transformer"), ("sarima", "lstm"), ("lstm", "transformer")]
    for ma, mb in pairs:
        for s in a.seeds:
            fa = _load(d, ma, 0 if ma == "sarima" else s)
            r = paired_bootstrap(fa, _load(d, mb, s), n=a.n_boot, seed=0)
            rows.append({"a": ma, "b": mb, "seed": s, "mae_diff_b_minus_a": r["diff"], "ci95_lo": r["lo"],
                         "ci95_hi": r["hi"], "days_b_better": r["b_wins"]})
    ens = []
    sar = _load(d, "sarima", 0)
    for m in ("lstm", "transformer"):
        e = seed_average(d, m, a.seeds)
        r = paired_bootstrap(sar, e, n=a.n_boot, seed=0)
        ens.append({"model": f"{m} ({len(a.seeds)}-seed mean)", "mae": float((e.y - e.yhat).abs().mean()),
                    "rmse": float(((e.y - e.yhat) ** 2).mean() ** 0.5), "vs_sarima": r["diff"],
                    "ci95_lo": r["lo"], "ci95_hi": r["hi"], "days_better": r["b_wins"]})
    text = "\n".join([
        "# Head-to-head (paired bootstrap over forecast days; negative = b has lower MAE)", "",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), "",
        "## Seed-averaged ensembles vs SARIMA", "",
        pd.DataFrame(ens).to_markdown(index=False, floatfmt=".4f"), "",
    ])
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
