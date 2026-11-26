import numpy as np
import pandas as pd
import pytest

from credit_default.evaluate import (
    aggregate,
    best_f1_threshold,
    compute_metrics,
    expected_calibration_error,
)


def test_ece_zero_when_probabilities_match_frequencies():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0)


def test_ece_known_value():
    # all predictions 0.9 but nobody defaults -> ECE = 0.9
    assert expected_calibration_error(np.zeros(20), np.full(20, 0.9)) == pytest.approx(0.9)


def test_ece_handles_probability_one():
    assert expected_calibration_error(np.ones(5), np.ones(5)) == pytest.approx(0.0)


def test_best_threshold_separates_perfectly():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.8
    assert compute_metrics(y, p, t)["f1"] == pytest.approx(1.0)


def test_aggregate_mean_std():
    df = pd.DataFrame({
        "seed": [0, 1], "model": ["m", "m"], "cv_pr_auc": [0.5, 0.7],
        **{k: [0.2, 0.4] for k in ["roc_auc", "pr_auc", "f1", "precision", "recall",
                                    "brier", "ece", "log_loss"]},
    })
    agg = aggregate(df)
    assert agg.loc["m", "pr_auc_mean"] == pytest.approx(0.3)
    assert agg.loc["m", "pr_auc_std"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert agg.loc["m", "n_seeds"] == 2
