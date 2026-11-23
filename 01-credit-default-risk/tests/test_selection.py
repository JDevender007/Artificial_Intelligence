import pandas as pd

from credit_default.experiment import select_final_family


def _rows(data: list[tuple[int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=["seed", "model", "cv_pr_auc"])


def test_selects_by_mean_over_seeds_not_per_seed():
    # Seed 0 prefers xgboost, seeds 1-2 prefer lightgbm; lightgbm wins on the mean.
    df = _rows([
        (0, "xgboost_tuned", 0.60), (0, "lightgbm_tuned", 0.59),
        (1, "xgboost_tuned", 0.55), (1, "lightgbm_tuned", 0.58),
        (2, "xgboost_tuned", 0.55), (2, "lightgbm_tuned", 0.58),
    ])
    assert select_final_family(df, [0, 1, 2]) == "lightgbm_tuned"


def test_ignores_dummy_calibrated_and_untuned_when_tuned_exist():
    df = _rows([
        (0, "dummy", 0.99), (0, "xgboost", 0.90), (0, "xgboost_tuned_calibrated", 0.95),
        (0, "logreg_tuned", 0.50), (0, "xgboost_tuned", 0.60),
    ])
    assert select_final_family(df, [0]) == "xgboost_tuned"


def test_falls_back_to_untuned_models():
    df = _rows([(0, "dummy", 0.2), (0, "logreg", 0.5), (0, "lightgbm", 0.55)])
    assert select_final_family(df, [0]) == "lightgbm"
