"""DVC pipeline stages: ``prepare`` -> ``train`` -> ``drift``. Each reads ``params.yaml`` and writes files only.

    python -m credops.pipeline prepare|train|drift [--params params.yaml]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from credops import data as D
from credops.drift import drift_table, drift_verdict, evidently_report, psi
from credops.model import build, metrics, oof_threshold
from credops.schema import MODEL_INPUTS, TARGET
from credops.simulate import SCENARIOS, scenario

log = logging.getLogger("credops")
SERVE_REQS = Path(__file__).resolve().parents[2] / "requirements-serve.txt"


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


# ---------------------------------------------------------------- prepare
def prepare(p: dict[str, Any]) -> dict[str, Any]:
    """Raw .xls -> cleaned, split parquet files plus a data card (rows, positive rate, SHA-256)."""
    xls = D.ensure_data(p["data"]["raw_dir"])
    df = D.clean(D.load_raw(xls))
    train, test = D.split(df, p["data"]["test_size"], p["data"]["split_seed"])
    out = Path(p["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    card = {"source_sha256": D.sha256(xls), "rows": len(df), "train_rows": len(train), "test_rows": len(test),
            "positive_rate": float(df[TARGET].mean()), "train_positive_rate": float(train[TARGET].mean()),
            "test_positive_rate": float(test[TARGET].mean())}
    _write_json(out / "data_card.json", card)
    log.info("prepare: %s", card)
    return card


# ---------------------------------------------------------------- train
def _mlflow(p: dict[str, Any]):
    if not p["mlflow"]["enabled"]:
        return None
    import os

    import mlflow

    # don't record the names of unrelated environment variables in the model artifact
    os.environ.setdefault("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", "false")
    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment"])
    return mlflow


def _trusted_types(model: Any) -> list[str]:
    """The classes skops must trust to reload ``model``: exactly the ones in the object we just trained ourselves."""
    import skops.io as sio

    return sorted(sio.get_untrusted_types(data=sio.dumps(model)))


def train(p: dict[str, Any]) -> dict[str, Any]:
    """Baseline vs candidate over several seeds on the fixed test split, a promotion gate, then export.

    Writes ``models/model.joblib`` + ``models/metadata.json`` (what the API serves), ``reports/metrics.json``
    (DVC metrics) and ``reports/results.md``. Every fit is logged to MLflow when enabled; the exported model is
    also registered there with the alias ``champion``.
    """
    proc = Path(p["data"]["processed_dir"])
    tr, te = pd.read_parquet(proc / "train.parquet"), pd.read_parquet(proc / "test.parquet")
    Xtr, ytr, Xte, yte = tr[MODEL_INPUTS], tr[TARGET], te[MODEL_INPUTS], te[TARGET].to_numpy()
    mlf, t = _mlflow(p), p["train"]
    rows, fitted = [], {}
    for kind in ("logreg", "lgbm"):
        for seed in t["seeds"]:
            t0 = time.perf_counter()
            model = build(kind, seed, t["params"].get(kind))
            thr = oof_threshold(model, Xtr, ytr, seed)
            model.fit(Xtr, ytr)
            m = metrics(yte, model.predict_proba(Xte)[:, 1], thr)
            m.update(threshold=thr, fit_s=time.perf_counter() - t0)
            rows.append({"model": kind, "seed": seed, **m})
            fitted[kind, seed] = model
            log.info("train %s seed %d: %s", kind, seed, {k: round(v, 4) for k, v in m.items()})
            if mlf:
                with mlf.start_run(run_name=f"{kind}-s{seed}"):
                    mlf.log_params({"model": kind, "seed": seed, **(t["params"].get(kind) or {})})
                    mlf.log_metrics({k: v for k, v in m.items()})
    res = pd.DataFrame(rows)
    agg = res.groupby("model").agg(["mean", "std"])
    base, cand = agg.loc["logreg"], agg.loc["lgbm"]
    gate = p["gate"]
    margin = cand[("pr_auc", "mean")] - base[("pr_auc", "mean")]
    std = cand[("pr_auc", "std")]
    if np.isnan(std):  # one seed: no spread estimate, so the gate reduces to "candidate is better"
        log.warning("gate: single seed, PR-AUC std unavailable; using 0")
        std = 0.0
    promote = bool(margin > gate["min_pr_auc_margin_std"] * std
                   and cand[("ece", "mean")] <= gate["max_ece"])
    champion = "lgbm" if promote else "logreg"
    seed0 = t["seeds"][0]
    model = fitted[champion, seed0]
    row = res[(res.model == champion) & (res.seed == seed0)].iloc[0]
    version = f"{champion}-{git_sha()}-{dt.datetime.now(dt.UTC):%Y%m%d%H%M}"
    meta = {"version": version, "kind": champion, "seed": seed0, "threshold": float(row.threshold),
            "test_metrics": {k: float(row[k]) for k in ("pr_auc", "roc_auc", "brier", "ece", "f1")},
            "features": MODEL_INPUTS, "trained_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "data_card": json.loads((proc / "data_card.json").read_text(encoding="utf-8")),
            "gate": {"promoted_candidate": promote, "pr_auc_margin": float(margin), **gate}}
    mdir = Path(p["paths"]["model_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, mdir / "model.joblib")
    _write_json(mdir / "metadata.json", meta)
    np.save(mdir / "reference_scores.npy", model.predict_proba(Xte)[:, 1])  # score-drift reference for the monitor
    if mlf:
        with mlf.start_run(run_name=f"export-{version}"):
            mlf.log_dict(meta, "metadata.json")
            info = mlf.sklearn.log_model(model, name="model", registered_model_name=p["mlflow"]["registered_model"],
                                         input_example=Xtr.head(2).astype(float),
                                         # MLflow 3.x saves sklearn with skops, which refuses unlisted classes
                                         skops_trusted_types=_trusted_types(model),
                                         pip_requirements=str(SERVE_REQS) if SERVE_REQS.exists() else None)
            mlf.MlflowClient().set_registered_model_alias(p["mlflow"]["registered_model"], "champion",
                                                          info.registered_model_version)
    summary = {k: {f"{m}_{s}": float(agg.loc[k][(m, s)]) for m in ("pr_auc", "roc_auc", "brier", "ece", "f1")
                   for s in ("mean", "std")} for k in ("logreg", "lgbm")}
    summary["champion"], summary["promoted_candidate"] = champion, promote
    rdir = Path(p["paths"]["report_dir"])
    _write_json(rdir / "metrics.json", summary)
    res.to_csv(rdir / "train_runs.csv", index=False)
    (rdir / "results.md").write_text(_results_md(agg, meta, len(t["seeds"])), encoding="utf-8")
    return summary


def _results_md(agg: pd.DataFrame, meta: dict[str, Any], n: int) -> str:
    cols = ("pr_auc", "roc_auc", "brier", "ece", "f1", "threshold", "fit_s")
    lines = [f"# Training results (held-out test, mean ± std over {n} seeds)", "",
             "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in ("logreg", "lgbm"):
        lines.append(f"| {k} | " + " | ".join(f"{agg.loc[k][(c, 'mean')]:.4f} ± {agg.loc[k][(c, 'std')]:.4f}"
                                              for c in cols) + " |")
    g = meta["gate"]
    lines += ["", f"Gate: candidate PR-AUC margin {g['pr_auc_margin']:.4f} must exceed "
              f"{g['min_pr_auc_margin_std']} × its seed std, and its ECE must be ≤ {g['max_ece']}. "
              f"Promoted: **{g['promoted_candidate']}**. Served model: `{meta['version']}` "
              f"(threshold {meta['threshold']:.4f}, chosen on out-of-fold train predictions)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- drift
def drift(p: dict[str, Any]) -> dict[str, Any]:
    """Run each simulated production scenario through the drift check and the served model."""
    proc, mdir, rdir = Path(p["data"]["processed_dir"]), Path(p["paths"]["model_dir"]), Path(p["paths"]["report_dir"])
    ref = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    model, meta = joblib.load(mdir / "model.joblib"), json.loads((mdir / "metadata.json").read_text("utf-8"))
    d = p["drift"]
    ref_scores = np.load(mdir / "reference_scores.npy")  # model scores on the clean test set
    out = {}
    for name in SCENARIOS:
        cur = scenario(test, name, seed=d["seed"])
        verdict = drift_verdict(drift_table(ref, cur, MODEL_INPUTS, d["psi_alert"], d["ks_alpha"]), d["share_alert"])
        prob = model.predict_proba(cur[MODEL_INPUTS])[:, 1]
        perf = metrics(cur[TARGET].to_numpy(), prob, meta["threshold"])
        out[name] = {"drift": verdict, "performance": perf, "default_rate": float(cur[TARGET].mean()),
                     "mean_probability": float(np.mean(prob)), "score_psi": psi(ref_scores, prob)}
        if d.get("evidently", True):
            evidently_report(ref[MODEL_INPUTS], cur[MODEL_INPUTS], rdir / "drift" / f"evidently_{name}.html")
    _write_json(rdir / "drift.json", out)
    lines = ["# Drift drills (simulated scenarios on the held-out test set)", "",
             "| scenario | drift alert | drifted cols | top PSI column | score PSI | default rate | mean P | PR-AUC | ECE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        top = r["drift"]["top"][0]
        lines.append(f"| {name} | {r['drift']['alert']} | {r['drift']['n_drifted']}/{r['drift']['n_columns']} | "
                     f"{top['column']} ({top['psi']:.3f}) | {r['score_psi']:.3f} | {r['default_rate']:.3f} | {r['mean_probability']:.3f} | "
                     f"{r['performance']['pr_auc']:.4f} | {r['performance']['ece']:.4f} |")
    (rdir / "drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


STAGES = {"prepare": prepare, "train": train, "drift": drift}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    STAGES[a.stage](load_params(a.params))


if __name__ == "__main__":
    main()
