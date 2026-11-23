"""Drift check on the API's own prediction log (what a scheduled monitoring job would run).

    python -m credops.monitor --log logs/predictions.jsonl [--min-rows 500]

Compares logged model inputs with the training reference and exits 1 on a drift alert, so a cron job or CI step
can page on it. Labels arrive months later in credit, so this watches inputs and score distribution only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from credops.drift import drift_table, drift_verdict, evidently_report, psi
from credops.pipeline import load_params
from credops.schema import MODEL_INPUTS

log = logging.getLogger("credops.monitor")


def check(log_path: str | Path, p: dict, min_rows: int = 500, html: str | Path | None = None) -> dict:
    cur = pd.read_json(log_path, lines=True)
    if len(cur) < min_rows:
        return {"status": "insufficient_data", "rows": len(cur), "min_rows": min_rows}
    ref = pd.read_parquet(Path(p["data"]["processed_dir"]) / "train.parquet")
    d = p["drift"]
    verdict = drift_verdict(drift_table(ref, cur, MODEL_INPUTS, d["psi_alert"], d["ks_alpha"]), d["share_alert"])
    out = {"status": "alert" if verdict["alert"] else "ok", "rows": len(cur), **verdict,
           "mean_probability": float(np.mean(cur["probability"]))}
    ref_scores = Path(p["paths"]["model_dir"]) / "reference_scores.npy"
    if ref_scores.exists():
        out["score_psi"] = psi(np.load(ref_scores), cur["probability"].to_numpy())
    if html:
        evidently_report(ref[MODEL_INPUTS], cur[MODEL_INPUTS], html)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="logs/predictions.jsonl")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--html")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    out = check(a.log, load_params(a.params), a.min_rows, a.html)
    print(json.dumps(out, indent=2, default=float))
    sys.exit(1 if out["status"] == "alert" else 0)


if __name__ == "__main__":
    main()
