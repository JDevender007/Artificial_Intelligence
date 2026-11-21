from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from textclf.baselines import fit_majority, fit_tfidf_lr
from textclf.bert import linear_warmup_lambda
from textclf.config import load_config
from textclf.experiment import run_experiment
from textclf.latency import measure
from textclf.utils import read_jsonl


def test_majority_predicts_most_frequent() -> None:
    y = np.array([2, 2, 1])
    p = fit_majority(["a", "b", "c"], y, 4).model.predict_proba(["x", "y"])
    assert p.shape == (2, 4) and (p.argmax(1) == 2).all() and p[:, [0, 3]].sum() == 0


def test_tfidf_learns_templates(splits) -> None:
    cfg = load_config(None, ["tfidf.c_grid=[1.0, 10.0]"]).tfidf
    r = fit_tfidf_lr(cfg, splits.train.text.tolist(), splits.train.label.to_numpy(), splits.val.text.tolist(),
                     splits.val.label.to_numpy(), 3, 0)
    assert r.info["C"] in (1.0, 10.0) and r.info["val_macro_f1"] == pytest.approx(1.0)
    probs = r.model.predict_proba(splits.test.text.tolist())
    assert np.allclose(probs.sum(1), 1) and (probs.argmax(1) == splits.test.label.to_numpy()).all()
    with pytest.raises(ValueError):
        r.model.to("cuda")


def test_warmup_schedule() -> None:
    f = linear_warmup_lambda(total_steps=10, warmup_steps=2)
    assert [round(f(s), 3) for s in (0, 1, 2, 6, 10)] == [0.5, 1.0, 1.0, 0.5, 0.0]


def test_latency_measure_counts() -> None:
    r = measure(lambda b: [len(t) for t in b], [f"t{i}" for i in range(10)], batch_size=4, warmup=1)
    assert r["requests"] == 2 and r["batch_size"] == 4 and r["ms_per_text"] >= 0  # partial chunk dropped


def test_end_to_end_resumable(tmp_path: Path, splits) -> None:
    cfg = load_config(None, [f"output_dir={tmp_path.as_posix()}", "models=[majority, tfidf_lr]", "seeds=[0, 1]",
                             "tfidf.c_grid=[5.0]", "n_bootstrap=20", "latency.n_texts=8", "latency.batch_sizes=[1]"])
    out = run_experiment(cfg, splits)
    rows = read_jsonl(out / "results.jsonl")
    assert [(r["model"], r["seed"]) for r in rows] == [("majority", 0), ("majority", 1), ("tfidf_lr", 0),
                                                         ("tfidf_lr", 1)]
    tf = [r for r in rows if r["model"] == "tfidf_lr"]
    assert all(r["macro_f1"] == pytest.approx(1.0) for r in tf)
    for f in ("results.md", "per_class.md", "error_analysis.md", "latency.md", "MODEL_CARD.md", "data_summary.md"):
        assert (out / f).stat().st_size > 0, f
    assert "| tfidf_lr | 1.000 ± 0.000 |" in (out / "results.md").read_text(encoding="utf-8")
    assert (out / "models" / "tfidf_lr.joblib").exists()
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Banking77" in card and "Gain over the TF-IDF baseline" not in card  # no BERT -> no comparison line
    run_experiment(cfg, splits)  # second call skips everything
    assert len(read_jsonl(out / "results.jsonl")) == 4
