"""End-to-end runner. Each (model, seed) row is appended to ``results.jsonl`` and its test probabilities are
saved, so an interrupted run resumes and every report is rebuilt from saved predictions by ``summarize``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import yaml

from .baselines import fit_majority, fit_tfidf_lr
from .config import ExperimentConfig
from .data import Splits, load_splits, split_stats, stats_markdown, word_count
from .latency import measure
from .metrics import (
    by_length,
    classification_metrics,
    ece,
    length_bucket_names,
    paired_bootstrap,
    per_class,
    top_confusions,
)
from .model_card import render_model_card
from .utils import aggregate, append_jsonl, get_logger, mean_std_markdown, read_jsonl, set_seed, write_json

log = get_logger(__name__)
METRICS = ["macro_f1", "accuracy", "ece"]


def _fit(name: str, cfg: ExperimentConfig, s: Splits, seed: int):
    tr_x, tr_y = s.train.text.tolist(), s.train.label.to_numpy()
    va_x, va_y = s.val.text.tolist(), s.val.label.to_numpy()
    if name == "majority":
        r = fit_majority(tr_x, tr_y, len(s.labels))
        return r.model, r.info
    if name == "tfidf_lr":
        r = fit_tfidf_lr(cfg.tfidf, tr_x, tr_y, va_x, va_y, len(s.labels), seed)
        return r.model, r.info
    from .bert import fit_bert  # lazy: torch + transformers

    return fit_bert(cfg.bert, tr_x, tr_y, va_x, va_y, s.labels, seed)


def _latency(name: str, predictor, cfg: ExperimentConfig, texts: list[str]) -> list[dict]:
    rows = []
    devices = ["cpu"] if name == "tfidf_lr" else cfg.latency.devices
    for dev in devices:
        if dev.startswith("cuda"):
            try:
                import torch
            except ImportError:
                continue
            if not torch.cuda.is_available():
                continue
        predictor.to(dev)
        for bs in cfg.latency.batch_sizes:
            r = measure(lambda b: predictor.predict_proba(b), texts[: cfg.latency.n_texts], bs, cfg.latency.warmup,
                        dev)
            rows.append({"model": name, "cpu_count": os.cpu_count(), **r})
            log.info("latency %s %s bs=%d: %.2f ms/text", name, dev, bs, r["ms_per_text"])
    if name == "distilbert":
        predictor.to("cuda" if any(r["device"].startswith("cuda") for r in rows) else "cpu")
    return rows


def run_experiment(cfg: ExperimentConfig, splits: Splits | None = None) -> Path:
    out = Path(cfg.output_dir)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    s = splits or load_splits(cfg.data)
    st = split_stats(s)
    write_json(st, out / "data_summary.json")
    (out / "data_summary.md").write_text(stats_markdown(st), encoding="utf-8")
    write_json(s.labels, out / "labels.json")
    np.save(out / "test_labels.npy", s.test.label.to_numpy())

    done = {(r["model"], r["seed"]) for r in read_jsonl(out / "results.jsonl")}
    test_x, test_y = s.test.text.tolist(), s.test.label.to_numpy()
    first = min(cfg.seeds)
    for name in cfg.models:
        for seed in cfg.seeds:
            if (name, seed) in done:
                log.info("skip %s seed %d (done)", name, seed)
                continue
            log.info("=== %s seed %d", name, seed)
            set_seed(seed)
            predictor, info = _fit(name, cfg, s, seed)
            probs = predictor.predict_proba(test_x)
            np.save(out / "preds" / f"{name}_s{seed}.npy", probs.astype(np.float32))
            row = {"model": name, "seed": seed, **classification_metrics(test_y, probs.argmax(1), len(s.labels)),
                   "ece": ece(probs, test_y)}
            row.update({k: v for k, v in info.items() if isinstance(v, int | float | str)})
            if seed == first and name != "majority":
                for r in _latency(name, predictor, cfg, test_x):
                    append_jsonl(r, out / "latency.jsonl")
                models_dir = out / "models"
                models_dir.mkdir(exist_ok=True)
                predictor.save(str(models_dir / (f"{name}.joblib" if name == "tfidf_lr" else name)))
            if "history" in info:
                write_json(info["history"], out / "history" / f"{name}_s{seed}.json")
            append_jsonl(row, out / "results.jsonl")
            del predictor
    summarize(cfg, s)
    return out


def _latency_markdown(rows: list[dict]) -> str:
    lines = ["| model | device | batch | ms/text | batch p50 ms | batch p95 ms | texts/s |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['device']} | {r['batch_size']} | {r['ms_per_text']:.3f} | "
                     f"{r['ms_per_batch_p50']:.2f} | {r['ms_per_batch_p95']:.2f} | {r['texts_per_s']:.0f} |")
    cpu = {r["cpu_count"] for r in rows}
    return "\n".join(lines) + f"\n\nCPU logical cores visible: {sorted(cpu)}. Batch-1 p95 is per-request latency."


def summarize(cfg: ExperimentConfig, s: Splits) -> None:
    """Rebuild results, per-class, error-analysis, latency and model-card files from saved predictions."""
    out = Path(cfg.output_dir)
    rows = read_jsonl(out / "results.jsonl")
    if not rows:
        return
    order = [m for m in cfg.models if any(r["model"] == m for r in rows)]
    rows = sorted(rows, key=lambda r: (order.index(r["model"]), r["seed"]))
    agg = aggregate(rows, METRICS)
    (out / "results.md").write_text(mean_std_markdown(agg, METRICS) + "\n\nTest set, mean ± std over seeds.",
                                    encoding="utf-8")
    agg.to_csv(out / "results.csv", index=False)

    y = s.test.label.to_numpy()
    labels, k = s.labels, len(s.labels)
    first = min(r["seed"] for r in rows)
    preds = {m: np.load(out / "preds" / f"{m}_s{first}.npy") for m in order}
    pc_all = {}
    for m in order:
        f1s = [[c["f1"] for c in per_class(y, np.load(p).argmax(1), labels)]
               for p in sorted((out / "preds").glob(f"{m}_s*.npy"))]
        pc_all[m] = np.mean(f1s, axis=0)
    lines = ["Test F1 per intent (mean over seeds), sorted by the last model", "",
             "| intent | support | " + " | ".join(order) + " |", "|---|---|" + "---|" * len(order)]
    support = np.bincount(y, minlength=k)
    for i in np.argsort(pc_all[order[-1]]):
        lines.append(f"| {labels[i]} | {support[i]} | " + " | ".join(f"{pc_all[m][i]:.3f}" for m in order) + " |")
    (out / "per_class.md").write_text("\n".join(lines), encoding="utf-8")

    ea = error_analysis(s, preds, cfg, first)
    write_json(ea, out / "error_analysis.json")
    (out / "error_analysis.md").write_text(error_markdown(ea), encoding="utf-8")

    lat = read_jsonl(out / "latency.jsonl")
    if lat:
        (out / "latency.md").write_text(_latency_markdown(lat), encoding="utf-8")
    st = json.loads((out / "data_summary.json").read_text(encoding="utf-8"))
    (out / "MODEL_CARD.md").write_text(render_model_card(cfg, agg, lat, st, ea), encoding="utf-8")


def error_analysis(s: Splits, preds: dict[str, np.ndarray], cfg: ExperimentConfig, seed: int,
                   n_examples: int = 15) -> dict:
    """Compare the main model (distilbert if present, else the last one) with the TF-IDF baseline."""
    y, labels, k = s.test.label.to_numpy(), s.labels, len(s.labels)
    models = [m for m in ("tfidf_lr", "distilbert") if m in preds] or list(preds)
    main, base = models[-1], models[0]
    wc = word_count(s.test.text)
    hard = {m: preds[m].argmax(1) for m in models}
    pc = {m: per_class(y, hard[m], labels) for m in models}
    worst = sorted(range(k), key=lambda i: pc[main][i]["f1"])[:15]
    ea: dict = {"seed": seed, "main": main, "baseline": base,
                "worst_classes": [{"intent": labels[i], "support": pc[main][i]["support"],
                                   **{f"{m}_f1": pc[m][i]["f1"] for m in models}} for i in worst],
                "baseline_better": sorted(
                    [{"intent": labels[i], f"{base}_f1": pc[base][i]["f1"], f"{main}_f1": pc[main][i]["f1"]}
                     for i in range(k) if pc[base][i]["f1"] > pc[main][i]["f1"] + 1e-9],
                    key=lambda r: r[f"{main}_f1"] - r[f"{base}_f1"]),
                "by_length": {m: by_length(wc, y, hard[m], cfg.data.length_edges, k) for m in models},
                "length_buckets": length_bucket_names(cfg.data.length_edges),
                "confusions": {m: top_confusions(y, hard[m], labels) for m in models}}
    if main != base:
        a, b = hard[base] == y, hard[main] == y
        ea["agreement"] = {"both_right": int((a & b).sum()), "only_baseline_right": int((a & ~b).sum()),
                           "only_main_right": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum())}
        ea["bootstrap"] = paired_bootstrap(y, hard[base], hard[main], k, cfg.n_bootstrap, seed)
    wrong = np.flatnonzero(hard[main] != y)
    pick = np.random.default_rng(seed).choice(wrong, size=min(n_examples, len(wrong)), replace=False)
    ea["examples"] = [{"text": s.test.text[i], "true": labels[y[i]], "predicted": labels[hard[main][i]],
                       "confidence": float(preds[main][i].max()), "words": int(wc[i])} for i in sorted(pick)]
    return ea


def error_markdown(ea: dict) -> str:
    main, base = ea["main"], ea["baseline"]
    models = list(ea["by_length"])
    L = [f"Main model `{main}` vs baseline `{base}`, seed {ea['seed']}, test set.", ""]
    if "bootstrap" in ea:
        b = ea["bootstrap"]
        L += [f"**Paired bootstrap** (macro-F1 {main} − {base}): {b['diff']:+.4f}, 95% CI "
              f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}], share of resamples with no gain {b['p_b_not_better']:.3f}.", ""]
        g = ea["agreement"]
        L += ["| both right | only baseline right | only main right | both wrong |", "|---|---|---|---|",
              f"| {g['both_right']} | {g['only_baseline_right']} | {g['only_main_right']} | {g['both_wrong']} |", ""]
    L += [f"### 15 weakest intents for `{main}`", "", "| intent | support | " + " | ".join(f"{m} F1" for m in models)
          + " |", "|---|---|" + "---|" * len(models)]
    for r in ea["worst_classes"]:
        L.append(f"| {r['intent']} | {r['support']} | " + " | ".join(f"{r[f'{m}_f1']:.3f}" for m in models) + " |")
    if main != base:
        L += ["", f"### Intents where `{base}` beats `{main}` ({len(ea['baseline_better'])})", "",
              f"| intent | {base} F1 | {main} F1 |", "|---|---|---|"]
        L += [f"| {r['intent']} | {r[f'{base}_f1']:.3f} | {r[f'{main}_f1']:.3f} |" for r in ea["baseline_better"]]
    L += ["", "### By query length (words)", "", "| words | n | " + " | ".join(f"{m} acc | {m} macro-F1" for m in models)
          + " |", "|---|---|" + "---|---|" * len(models)]
    for i, name in enumerate(ea["length_buckets"]):
        cells = []
        for m in models:
            r = ea["by_length"][m][i]
            cells += [f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}"]
        L.append(f"| {name} | {ea['by_length'][models[0]][i]['n']} | " + " | ".join(cells) + " |")
    for m in models:
        L += ["", f"### Top confusions: `{m}`", "", "| true | predicted | count |", "|---|---|---|"]
        L += [f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in ea["confusions"][m]]
    L += ["", f"### Random misclassified test queries (`{main}`)", "", "| text | true | predicted | conf |",
          "|---|---|---|---|"]
    L += [f"| {e['text'].replace('|', '/')} | {e['true']} | {e['predicted']} | {e['confidence']:.2f} |"
          for e in ea["examples"]]
    return "\n".join(L)
