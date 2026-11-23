"""Majority-class and TF-IDF + logistic-regression baselines (scikit-learn only)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import TfidfConfig
from .metrics import classification_metrics
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class FitResult:
    model: object
    info: dict = field(default_factory=dict)  # selected hyper-parameters, val score, fit time


class SklearnPredictor:
    """Common interface: ``predict_proba(list[str]) -> (n, n_labels)`` with columns in label-id order."""

    def __init__(self, model, n_labels: int) -> None:
        self.model, self.n_labels = model, n_labels

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        p = self.model.predict_proba(list(texts))
        out = np.zeros((len(texts), self.n_labels))
        out[:, self.model.classes_] = p  # classes absent from training get probability 0
        return out

    def to(self, device: str) -> SklearnPredictor:  # same interface as the BERT predictor
        if device != "cpu":
            raise ValueError("scikit-learn models run on CPU only")
        return self

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)


def fit_majority(train_texts: list[str], train_y: np.ndarray, n_labels: int) -> FitResult:
    m = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train_y), 1)), train_y)

    class _Wrap:  # DummyClassifier ignores X; accept raw texts like the other models
        classes_ = m.classes_

        def predict_proba(self, texts: list[str]) -> np.ndarray:
            return m.predict_proba(np.zeros((len(texts), 1)))

    return FitResult(SklearnPredictor(_Wrap(), n_labels), {"strategy": "most_frequent"})


def build_tfidf_lr(cfg: TfidfConfig, c: float, seed: int) -> Pipeline:
    parts = [("word", TfidfVectorizer(ngram_range=tuple(cfg.word_ngrams), min_df=cfg.min_df, sublinear_tf=True,
                                      lowercase=True))]
    if cfg.char_ngrams:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg.char_ngrams),
                                              min_df=cfg.min_df, sublinear_tf=True, lowercase=True)))
    # sklearn >= 1.8: the default lbfgs solver is multinomial; ``multi_class``/``penalty`` are not passed
    return Pipeline([("features", FeatureUnion(parts)),
                     ("clf", LogisticRegression(C=c, max_iter=cfg.max_iter, random_state=seed))])


def fit_tfidf_lr(cfg: TfidfConfig, train_texts: list[str], train_y: np.ndarray, val_texts: list[str],
                 val_y: np.ndarray, n_labels: int, seed: int) -> FitResult:
    """Grid-search C on the validation macro-F1, keep the best model (fit on train only)."""
    best, best_f1, scores = None, -1.0, {}
    t0 = time.perf_counter()
    for c in cfg.c_grid:
        pipe = build_tfidf_lr(cfg, c, seed).fit(train_texts, train_y)
        f1 = classification_metrics(val_y, pipe.predict(val_texts), n_labels)["macro_f1"]
        scores[str(c)] = f1
        log.info("tfidf_lr C=%s val macro-F1 %.4f", c, f1)
        if f1 > best_f1:
            best, best_f1, best_c = pipe, f1, c
    if len(cfg.c_grid) > 1 and best_c in (min(cfg.c_grid), max(cfg.c_grid)):
        log.warning("tfidf_lr best C=%s is at the edge of c_grid %s; consider widening it", best_c, cfg.c_grid)
    n_feat = len(best.named_steps["features"].get_feature_names_out())
    return FitResult(SklearnPredictor(best, n_labels),
                     {"C": best_c, "val_macro_f1": best_f1, "val_by_C": scores, "n_features": n_feat,
                      "fit_s": time.perf_counter() - t0})
