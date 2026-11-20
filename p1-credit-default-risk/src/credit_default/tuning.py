"""Optuna hyper-parameter search on TRAIN-ONLY stratified CV (test set never seen)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from credit_default.models import build_pipeline


def search_space(trial: optuna.Trial, name: str) -> dict[str, Any]:
    """Model-specific search space."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    if name == "xgboost":
        return {
            **common,
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if name == "lightgbm":
        return {
            **common,
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample_freq": 1,
        }
    raise ValueError(f"No search space for {name!r}")


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_trials: int,
    cv_folds: int,
    pos_weight: float | None,
    drop_protected: bool,
    engineered: bool,
    storage_dir: Path,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    """Maximise mean CV PR-AUC. The study is persisted in SQLite, so a killed
    Kaggle/Colab session resumes where it stopped instead of starting over."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(storage_dir / 'optuna.db').resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"{name}_seed{seed}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial, name)
        pipe = build_pipeline(name, params, seed, pos_weight, drop_protected, engineered)
        return float(cross_val_score(pipe, X, y, cv=cv, scoring="average_precision").mean())

    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - done)
    if remaining:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=remaining, timeout=timeout_s)
    params = dict(study.best_params)
    if name == "lightgbm":
        params["subsample_freq"] = 1
    return params
