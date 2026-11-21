"""Top-k intents for queries, from a saved model of a finished run.

    python scripts/predict.py --run outputs/banking77 --model tfidf_lr "my card still hasn't arrived"
    python scripts/predict.py --run outputs/banking77 --model distilbert --threshold 0.5 "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_predictor(run: Path, model: str, n_labels: int):
    if model == "tfidf_lr":
        import joblib

        from textclf.baselines import SklearnPredictor

        return SklearnPredictor(joblib.load(run / "models" / "tfidf_lr.joblib"), n_labels)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from textclf.bert import BertPredictor
    from textclf.config import load_config

    cfg = load_config(run / "config.yaml")
    path = run / "models" / "distilbert"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForSequenceClassification.from_pretrained(path).to(dev)
    return BertPredictor(m, AutoTokenizer.from_pretrained(path), cfg.bert, dev)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--run", default="outputs/banking77", type=Path)
    ap.add_argument("--model", default="tfidf_lr", choices=["tfidf_lr", "distilbert"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.0, help="below this top-1 confidence, answer 'unsure'")
    a = ap.parse_args()
    labels = json.loads((a.run / "labels.json").read_text(encoding="utf-8"))
    probs = load_predictor(a.run, a.model, len(labels)).predict_proba(a.texts)
    for text, p in zip(a.texts, probs, strict=True):
        top = np.argsort(p)[::-1][: a.k]
        verdict = labels[top[0]] if p[top[0]] >= a.threshold else "unsure -> hand to a human"
        print(f"{text!r} -> {verdict}")
        for i in top:
            print(f"    {labels[i]:<40} {p[i]:.3f}")


if __name__ == "__main__":
    main()
