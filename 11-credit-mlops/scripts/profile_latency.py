"""In-process latency profile of the exported model (no HTTP): where a single prediction spends its time.

    python scripts/profile_latency.py [--model-dir models] [--reps 200] > reports/latency_profile.md
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credops.schema import MODEL_INPUTS
from credops.synthetic import synthetic_frame


def timeit(f: Callable[[], object], reps: int) -> tuple[float, float]:
    """Median and p95 wall time of ``f`` in milliseconds."""
    f()  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    a = np.array(ts) * 1000
    return float(np.median(a)), float(np.percentile(a, 95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--test", default="data/processed/test.parquet")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    model = joblib.load(Path(a.model_dir) / "model.joblib")
    X = pd.read_parquet(a.test)[MODEL_INPUTS] if Path(a.test).exists() else synthetic_frame(2000, seed=1)[MODEL_INPUTS]
    feat, clf = model.named_steps["features"], model.named_steps["clf"]
    print("| step | rows | median ms | p95 ms |\n|---|---|---|---|")
    for n in (1, 50, 1000):
        x = X.iloc[:n]
        f = feat.transform(x)
        rows = [("full pipeline", lambda x=x: model.predict_proba(x)),
                ("feature engineering (pandas)", lambda x=x: feat.transform(x)),
                ("classifier", lambda f=f: clf.predict_proba(f))]
        for name, fn in rows:
            med, p95 = timeit(fn, a.reps if n < 1000 else max(20, a.reps // 10))
            print(f"| {name} | {n} | {med:.2f} | {p95:.2f} |")
    inner = getattr(clf, "calibrated_classifiers_", None)
    if inner:
        print(f"\nClassifier = {len(inner)} calibrated LightGBM folds, averaged; "
              f"n_jobs={inner[0].estimator.n_jobs} per booster.")


if __name__ == "__main__":
    main()
