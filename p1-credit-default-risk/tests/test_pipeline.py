import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from credit_default.config import load_config
from credit_default.data import get_xy
from credit_default.leakage import duplicate_overlap, run_leakage_checks, single_feature_auc
from credit_default.models import MODEL_NAMES, build_pipeline


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("drop_protected", [True, False])
def test_every_pipeline_fits_and_predicts(synthetic, name, drop_protected, tmp_path):
    X, y = get_xy(synthetic)
    pipe = build_pipeline(name, {"n_estimators": 20} if name in ("xgboost", "lightgbm") else {},
                          seed=0, pos_weight=3.0, drop_protected=drop_protected)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1))
    joblib.dump(pipe, tmp_path / "m.joblib")  # must be picklable (no lambdas)
    np.testing.assert_allclose(joblib.load(tmp_path / "m.joblib").predict_proba(X)[:, 1], p)


def test_pipeline_learns_signal_in_cv(synthetic):
    X, y = get_xy(synthetic)
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    auc = cross_val_score(build_pipeline("logreg"), X, y, cv=cv, scoring="roc_auc").mean()
    assert auc > 0.7  # synthetic target depends on PAY_1


def test_leakage_checks_flag_target_copy(synthetic):
    X, y = get_xy(synthetic)
    X = X.assign(leak=y * 10)
    assert "leak" in single_feature_auc(X, y).index[:1]
    report = run_leakage_checks(X, y, X.iloc[:50], y.iloc[:50])
    assert not report["passed"] and "leak" in report["suspicious_features"]


def test_duplicate_overlap_counts_rows(synthetic):
    X, _ = get_xy(synthetic)
    assert duplicate_overlap(X.iloc[:100], X.iloc[90:120]) == 10


def test_config_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("seeds: [0]\ntuning:\n  n_trials: 5\n")
    cfg = load_config(cfg_file, ["tuning.n_trials=7", "features.drop_protected=false"])
    assert cfg.seeds == [0] and cfg.tuning.n_trials == 7
    assert cfg.features.drop_protected is False
    with pytest.raises(KeyError):
        load_config(cfg_file, ["not_a_key=1"])
