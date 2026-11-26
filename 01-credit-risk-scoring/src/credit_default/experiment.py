"""End-to-end experiment: data -> leakage checks -> baselines -> tuning -> calibration
-> mean ± std results over seeds -> plots -> SHAP.

Resumable: every (seed, model) row is appended to ``results_per_seed.csv`` as soon as it
finishes and Optuna studies live in SQLite, so re-running after a crash skips done work.
"""

from __future__ import annotations

import json
import time
from logging import Logger
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit_default import plots
from credit_default.config import ExperimentConfig
from credit_default.data import clean, ensure_data, get_xy, load_raw, sha256, split
from credit_default.evaluate import aggregate, best_f1_threshold, compute_metrics, to_markdown
from credit_default.explain import shap_summary
from credit_default.leakage import run_leakage_checks
from credit_default.models import TREE_MODELS, build_pipeline
from credit_default.tuning import tune
from credit_default.utils import set_seed


def prepare_data(cfg: ExperimentConfig) -> dict[str, Any]:
    """Load, clean and split. Deterministic given ``cfg.data``."""
    xls = ensure_data(cfg.data.raw_dir)
    df = clean(load_raw(xls))
    train, test = split(df, cfg.data.test_size, cfg.data.split_seed)
    X_tr, y_tr = get_xy(train)
    X_te, y_te = get_xy(test)
    return {"df": df, "xls": xls, "train": train, "test": test,
            "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te}


def _append_row(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _fit_and_score(
    name: str, estimator: Any, seed: int, d: dict[str, Any], cv: StratifiedKFold,
    cfg: ExperimentConfig, out: Path, logger: Logger,
) -> dict[str, Any]:
    """OOF predictions on train pick the F1 threshold; then refit on all train and
    score the untouched test set once."""
    t0 = time.perf_counter()
    oof = cross_val_predict(estimator, d["X_tr"], d["y_tr"], cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(d["y_tr"].to_numpy(), oof)
    estimator.fit(d["X_tr"], d["y_tr"])
    p_test = estimator.predict_proba(d["X_te"])[:, 1]
    row = {
        "seed": seed,
        "model": name,
        "cv_pr_auc": float(average_precision_score(d["y_tr"], oof)),
        "cv_roc_auc": float(roc_auc_score(d["y_tr"], oof)),
        **compute_metrics(d["y_te"].to_numpy(), p_test, threshold, cfg.ece_bins),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    np.save(out / "preds" / f"seed{seed}_{name}.npy", p_test)
    joblib.dump(estimator, out / "models" / f"seed{seed}_{name}.joblib")
    _append_row(out / "results_per_seed.csv", row)
    logger.info(
        "seed=%d %-24s cv_pr_auc=%.4f test_pr_auc=%.4f test_roc_auc=%.4f f1=%.4f ece=%.4f (%.0fs)",
        seed, name, row["cv_pr_auc"], row["pr_auc"], row["roc_auc"], row["f1"], row["ece"],
        row["seconds"],
    )
    return row


def select_final_family(per_seed: pd.DataFrame, seeds: list[int], logger: Logger | None = None) -> str:
    """Pick ONE model to calibrate for every seed: the highest mean OOF PR-AUC over seeds.

    Selecting per seed would let the "final model" switch family between seeds, so its
    mean ± std would mix two different models. Only train-CV scores are used; the test
    set plays no part in selection. Tuned models are preferred over untuned ones.
    """
    rows = per_seed[per_seed["seed"].isin(seeds) & (per_seed["model"] != "dummy")]
    rows = rows[~rows["model"].str.endswith("_calibrated")]
    tuned = rows[rows["model"].str.endswith("_tuned")]
    pool = tuned if not tuned.empty else rows
    means = pool.groupby("model")["cv_pr_auc"].mean().sort_values(ascending=False)
    if logger is not None:
        logger.info("Mean CV PR-AUC over seeds (selection for calibration):\n%s", means.to_string())
    return str(means.index[0])


def run_experiment(cfg: ExperimentConfig, logger: Logger) -> pd.DataFrame:
    """Run every seed/model in ``cfg`` and write all artefacts to ``cfg.output_dir``."""
    out = Path(cfg.output_dir)
    for sub in ("models", "preds", "figures", "params"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    d = prepare_data(cfg)
    summary = {
        "n_rows": len(d["df"]),
        "n_raw_features": d["X_tr"].shape[1],
        "positive_rate": round(float(d["df"]["default"].mean()), 4),
        "n_train": len(d["X_tr"]),
        "n_test": len(d["X_te"]),
        "data_sha256": sha256(d["xls"]),
    }
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Data: %s", summary)

    leak = run_leakage_checks(d["X_tr"], d["y_tr"], d["X_te"], d["y_te"])
    (out / "leakage_report.json").write_text(json.dumps(leak, indent=2))
    logger.info("Leakage checks passed=%s, top single-feature AUCs=%s",
                leak["passed"], leak["top_single_feature_auc"])
    if not leak["passed"]:
        raise RuntimeError(f"Leakage check failed: {leak}")

    y_tr = d["y_tr"]
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum()) if cfg.imbalance == "class_weight" else None
    feat_kw = {"drop_protected": cfg.features.drop_protected, "engineered": cfg.features.engineered}

    per_seed_path = out / "results_per_seed.csv"
    done: set[tuple[int, str]] = set()
    if per_seed_path.exists():
        prev = pd.read_csv(per_seed_path)
        done = set(zip(prev["seed"].astype(int), prev["model"], strict=True))
        logger.info("Resuming: %d (seed, model) results already on disk", len(done))

    all_candidates: dict[int, dict[str, tuple[str, dict[str, Any]]]] = {}
    for seed in cfg.seeds:
        set_seed(seed)
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
        candidates: dict[str, tuple[str, dict[str, Any]]] = {m: (m, {}) for m in cfg.models}
        all_candidates[seed] = candidates
        for m in cfg.tune:
            params_path = out / "params" / f"seed{seed}_{m}.json"
            if params_path.exists():
                params = json.loads(params_path.read_text())
            else:
                logger.info("seed=%d tuning %s (%d trials)...", seed, m, cfg.tuning.n_trials)
                params = tune(
                    m, d["X_tr"], y_tr, seed=seed, n_trials=cfg.tuning.n_trials,
                    cv_folds=cfg.cv_folds, pos_weight=pos_weight, storage_dir=out,
                    timeout_s=cfg.tuning.timeout_s, **feat_kw,
                )
                params_path.write_text(json.dumps(params, indent=2))
            candidates[f"{m}_tuned"] = (m, params)

        for name, (base, params) in candidates.items():
            if (seed, name) in done:
                continue
            pipe = build_pipeline(base, params, seed, pos_weight, **feat_kw)
            _fit_and_score(name, pipe, seed, d, cv, cfg, out, logger)

    final = None
    if cfg.calibration != "none":
        best = select_final_family(pd.read_csv(per_seed_path), cfg.seeds, logger)
        final = f"{best}_calibrated"
        for seed in cfg.seeds:
            if (seed, final) in done:
                continue
            set_seed(seed)
            cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed)
            base, params = all_candidates[seed][best]
            cal = CalibratedClassifierCV(
                build_pipeline(base, params, seed, pos_weight, **feat_kw),
                method=cfg.calibration,
                cv=StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=seed),
            )
            _fit_and_score(final, cal, seed, d, cv, cfg, out, logger)

    per_seed = pd.read_csv(per_seed_path)
    per_seed = per_seed[per_seed["seed"].isin(cfg.seeds)]
    # Calibrated rows of a family that is no longer selected (e.g. from an older version
    # of the selection rule) stay in results_per_seed.csv but are not reported.
    stale = per_seed["model"].str.endswith("_calibrated") & (per_seed["model"] != final)
    per_seed = per_seed[~stale]
    agg = aggregate(per_seed)
    agg.to_csv(out / "results.csv")
    baseline = "logreg_tuned" if "logreg_tuned" in agg.index else "logreg"
    header = (
        f"# Results: {cfg.name}\n\n"
        f"Test set: {summary['n_test']} rows (positive rate {summary['positive_rate']:.4f}), "
        f"identical for every seed. Values are mean ± std over seeds {cfg.seeds}. "
        "`cv_pr_auc` is out-of-fold on train; all other columns are on the held-out test set. "
        "F1/precision/recall use the threshold that maximised OOF F1 on train.\n\n"
    )
    (out / "results.md").write_text(header + to_markdown(agg, baseline) + "\n", encoding="utf-8")
    logger.info("\n%s", to_markdown(agg, baseline))

    _make_figures(cfg, per_seed, d, out, baseline, logger)
    return agg


def _make_figures(
    cfg: ExperimentConfig, per_seed: pd.DataFrame, d: dict[str, Any], out: Path,
    baseline: str, logger: Logger,
) -> None:
    """ROC/PR, reliability and SHAP plots for the first seed."""
    seed = cfg.seeds[0]
    rows = per_seed[per_seed["seed"] == seed].set_index("model")
    calibrated = [m for m in rows.index if m.endswith("_calibrated")]
    final = calibrated[0] if calibrated else rows["cv_pr_auc"].idxmax()
    best_uncal = final.removesuffix("_calibrated")
    y = d["y_te"].to_numpy()

    def load(m: str) -> np.ndarray:
        return np.load(out / "preds" / f"seed{seed}_{m}.npy")

    shown = [m for m in dict.fromkeys([baseline, best_uncal, final]) if m in rows.index]
    plots.roc_pr_curves(y, {m: load(m) for m in shown}, out / "figures" / "roc_pr.png")
    plots.reliability_diagram(
        y, {m: load(m) for m in shown}, out / "figures" / "reliability.png", cfg.ece_bins
    )
    (out / "final_model.json").write_text(json.dumps({
        "seed": seed, "model": final, "threshold": float(rows.loc[final, "threshold"]),
        "uncalibrated": best_uncal,
    }, indent=2))

    base = best_uncal.removesuffix("_tuned")
    if base in TREE_MODELS:
        pipe = joblib.load(out / "models" / f"seed{seed}_{best_uncal}.joblib")
        imp = shap_summary(pipe, d["X_te"], out / "figures", cfg.shap_samples, seed)
        logger.info("Top SHAP features:\n%s", imp.head(10).to_string(index=False))
