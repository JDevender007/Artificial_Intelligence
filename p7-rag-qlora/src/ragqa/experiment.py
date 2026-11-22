"""End-to-end experiment: data -> retrieval ablation -> chosen setup -> base variants -> QLoRA seeds -> reports.

Every stage writes its outputs under ``output_dir`` and is skipped when they already exist, so a run that dies
(Kaggle session limit, OOM) resumes where it stopped.
"""

from __future__ import annotations

import gc
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ragqa.bm25 import BM25
from ragqa.chunking import Chunk, SupportIndex, make_chunks
from ragqa.config import BASE_VARIANTS, FT_VARIANT, ExperimentConfig
from ragqa.data import Dataset, Question, build_dataset
from ragqa.generate import Backends, Generator, SFTExample, hf_backends
from ragqa.metrics import (
    bootstrap_ci,
    contains,
    context_precision,
    mrr_at_k,
    paired_bootstrap,
    recall_at_k,
    squad_em,
    squad_f1,
    summarize,
)
from ragqa.prompts import ABSTAIN, build_messages, parse_reply, target_text
from ragqa.retrieve import Retriever
from ragqa.text import question_type
from ragqa.utils import aggregate, append_jsonl, get_logger, mean_std_markdown, read_jsonl, write_json

log = get_logger()
RESULT_METRICS = ["em", "f1", "has_ans_f1", "no_ans_acc", "false_abstain", "context_recall", "faithfulness",
                  "cite_rate", "cite_supports_answer", "cite_gold", "ms_per_q"]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------- retrieval ablation


def _max_k(cfg: ExperimentConfig) -> int:
    return max([*cfg.retrieval.ks, 10])


def retrieval_rows(retriever: Retriever, support: SupportIndex, questions: list[Question], method: str,
                   ks: list[int], k_search: int) -> dict[str, Any]:
    """Recall@k, MRR@10 and latency over the answerable questions."""
    qs = [q for q in questions if not q.is_impossible]
    t0 = time.perf_counter()
    rankings = retriever.search([q.text for q in qs], method, k_search)
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(qs))
    golds = [support.gold(q) for q in qs]
    row: dict[str, Any] = {"n": len(qs), "ms_per_query": ms}
    for k in ks:
        row[f"recall@{k}"] = float(np.mean([recall_at_k(r, g, k) for r, g in zip(rankings, golds, strict=True)]))
    row["mrr@10"] = float(np.mean([mrr_at_k(r, g, 10) for r, g in zip(rankings, golds, strict=True)]))
    return row


def run_retrieval(cfg: ExperimentConfig, ds: Dataset, backends: Backends, out: Path) -> dict[str, Any]:
    """Ablation over chunkers x methods on val and test; choose by val recall@k_context (then MRR).

    Also writes the chosen setup's top-k_context passages for every test question (``contexts.json``).
    """
    r = cfg.retrieval
    choice_path = out / "retrieval_choice.json"
    if choice_path.exists() and (out / "contexts.json").exists():
        return json.loads(choice_path.read_text(encoding="utf-8"))
    path = out / "retrieval.jsonl"
    done = {(x["chunker"], x["method"], x["split"]) for x in read_jsonl(path)}
    need_dense = any(m != "bm25" for m in r.methods)
    embedder = backends.embedder() if need_dense else None
    reranker = backends.reranker() if "hybrid_rerank" in r.methods else None
    retrievers: dict[str, Retriever] = {}
    for chunker in r.chunkers:
        chunks = make_chunks(ds.dev_articles, chunker, r.window_words, r.window_overlap)
        ret = Retriever(chunks, embedder, reranker, r.n_candidates, r.rrf_k)
        retrievers[chunker] = ret
        support = SupportIndex(chunks)
        for split, qs in (("val", ds.val_questions), ("test", ds.test_questions)):
            for method in r.methods:
                if (chunker, method, split) in done:
                    continue
                if method != "bm25":
                    ret.doc_embeddings()  # encode the corpus once, outside the per-query timing
                row = {"chunker": chunker, "method": method, "split": split, "n_chunks": len(chunks),
                       **retrieval_rows(ret, support, qs, method, r.ks, _max_k(cfg))}
                log.info("retrieval %s/%s/%s recall@%d=%.3f", chunker, method, split, r.k_context,
                         row[f"recall@{r.k_context}"])
                append_jsonl(row, path)
    rows = read_jsonl(path)
    val = [x for x in rows if x["split"] == "val" and x["chunker"] in r.chunkers and x["method"] in r.methods]
    best = max(val, key=lambda x: (x[f"recall@{r.k_context}"], x["mrr@10"]))
    choice = {"chunker": best["chunker"], "method": best["method"], "k_context": r.k_context,
              "val_recall": best[f"recall@{r.k_context}"], "val_mrr@10": best["mrr@10"]}
    write_json(choice, choice_path)
    (out / "retrieval.md").write_text(retrieval_markdown(rows, r.ks), encoding="utf-8")
    ret = retrievers[choice["chunker"]]
    rankings = ret.search([q.text for q in ds.test_questions], choice["method"], r.k_context)
    write_json({"chunker": choice["chunker"], "method": choice["method"],
                "contexts": {q.qid: rk for q, rk in zip(ds.test_questions, rankings, strict=True)}},
               out / "contexts.json")
    del retrievers, ret, embedder, reranker
    free_memory()
    return choice


def retrieval_markdown(rows: list[dict[str, Any]], ks: list[int]) -> str:
    cols = [f"recall@{k}" for k in ks] + ["mrr@10", "ms_per_query"]
    lines = []
    for split in ("val", "test"):
        sub = [x for x in rows if x["split"] == split]
        if not sub:
            continue
        lines += [f"### {split} (answerable questions, n={sub[0]['n']})", "",
                  "| chunker | method | chunks | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
        for x in sub:
            cells = [f"{x[c]:.1f}" if c == "ms_per_query" else f"{x[c]:.3f}" for c in cols]
            lines.append(f"| {x['chunker']} | {x['method']} | {x['n_chunks']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- generation and scoring


def score_record(q: Question, reply: str, ctx: list[Chunk] | None, gold: set[int]) -> dict[str, Any]:
    """Per-question record with every field :func:`ragqa.metrics.summarize` reads."""
    p = parse_reply(reply)
    rec: dict[str, Any] = {
        "qid": q.qid, "question": q.text, "qtype": question_type(q.text), "golds": list(q.answers),
        "is_impossible": q.is_impossible, "reply": reply, "pred": p.answer, "abstained": p.abstained,
        "citations": list(p.citations), "em": squad_em(p.answer, q.answers), "f1": squad_f1(p.answer, q.answers),
        "context": None,
    }
    if ctx is not None:
        ids = [c.cid for c in ctx]
        cited = [ctx[i - 1] for i in p.citations if 1 <= i <= len(ctx)]
        rec.update({
            "context": ids, "gold_chunks": sorted(gold),
            "retrieval_hit": any(c in gold for c in ids), "context_precision": context_precision(ids, gold),
            "grounded": (not p.abstained) and contains(" ".join(c.text for c in ctx), p.answer),
            "cite_supports_answer": any(contains(c.text, p.answer) for c in cited),
            "cite_gold": any(c.cid in gold for c in cited),
        })
    return rec


def generate_variant(variant: str, questions: list[Question], generator: Generator, chunks: list[Chunk] | None,
                     contexts: dict[str, list[int]] | None, support: SupportIndex | None) -> tuple[list[dict], float]:
    rag = variant.startswith("rag")
    ctxs = [[chunks[i] for i in contexts[q.qid]] if rag else None for q in questions]  # type: ignore[index]
    msgs = [build_messages(variant, q.text, c) for q, c in zip(questions, ctxs, strict=True)]
    t0 = time.perf_counter()
    replies = generator.generate(msgs)
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(questions))
    recs = [score_record(q, rep, c, support.gold(q) if (rag and support) else set())
            for q, rep, c in zip(questions, replies, ctxs, strict=True)]
    return recs, ms


def _write_predictions(recs: list[dict[str, Any]], ms: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    for r in recs:
        append_jsonl({**r, "ms_per_q": ms}, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------- fine-tuning data


def sft_examples(mode: str, cfg: ExperimentConfig, ds: Dataset, chunker: str, seed: int) -> list[SFTExample]:
    """closed: question -> answer. rag (RAFT-style): gold passage + BM25 hard negatives -> ``answer [n]``;
    a share ``p_drop_gold`` of answerable examples loses every passage holding the answer -> ``unanswerable``."""
    rng = np.random.default_rng(1000 + seed)
    if mode == "closed":
        return [{"messages": build_messages("ft_closed", q.text),
                 "target": target_text(q.answers[0] if q.answers else None)} for q in ds.train_questions]
    r = cfg.retrieval
    chunks = make_chunks(ds.train_articles, chunker, r.window_words, r.window_overlap)
    support = SupportIndex(chunks)
    bm = BM25().fit([c.index_text for c in chunks])
    k = r.k_context
    out = []
    for q in ds.train_questions:
        cand = bm.search(q.text, 4 * k)
        gold = support.gold(q)
        if q.is_impossible:
            ctx, target = cand[:k], target_text(None)
        else:
            negs = [c for c in cand if c not in gold and not any(contains(chunks[c].text, a) for a in q.answers)]
            if rng.random() < cfg.ft.p_drop_gold:
                ctx, target = negs[:k], target_text(None)
            else:
                g = next((c for c in cand if c in gold), min(gold))
                ctx = [g, *negs[:k - 1]]
                rng.shuffle(ctx)
                target = target_text(q.answers[0], ctx.index(g) + 1)
        out.append({"messages": build_messages("rag_ft", q.text, [chunks[c] for c in ctx]), "target": target})
    return out


# ---------------------------------------------------------------- reports


_SEEDED = re.compile(r"^(.*)_s(\d+)$")


def _variant_of(name: str) -> str:
    """``rag_ft_s1`` -> ``rag_ft``; unseeded names are returned unchanged."""
    m = _SEEDED.match(name)
    return m.group(1) if m else name


def collect_results(out: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted((out / "predictions").glob("*.jsonl")):
        recs = read_jsonl(p)
        if not recs:
            continue
        variant = _variant_of(p.stem)
        m = _SEEDED.match(p.stem)
        seed = int(m.group(2)) if m else None
        rows.append({"variant": variant, "seed": seed, **summarize(recs), "ms_per_q": recs[0]["ms_per_q"]})
    return rows


VARIANT_ORDER = ["base", "prompted", "ft_closed", "rag", "rag_nf4", "rag_ft"]


def results_markdown(rows: list[dict[str, Any]]) -> str:
    rows = sorted(rows, key=lambda x: (VARIANT_ORDER.index(x["variant"]) if x["variant"] in VARIANT_ORDER else 99,
                                       x["seed"] or 0))
    metrics = [m for m in RESULT_METRICS if any(m in r for r in rows)]
    filled = [{m: r.get(m, float("nan")) for m in metrics} | {"variant": r["variant"]} for r in rows]
    agg = aggregate(filled, metrics, by="variant")
    return mean_std_markdown(agg, metrics, by="variant").replace("| nan ", "| — ")


def _per_question(out: Path, variant: str) -> tuple[list[str], np.ndarray] | None:
    """Per-question F1 of a variant, averaged over its seeds (seed files are aligned by qid)."""
    files = sorted((out / "predictions").glob(f"{variant}.jsonl")) + sorted(
        (out / "predictions").glob(f"{variant}_s*.jsonl"))
    files = [f for f in files if _variant_of(f.stem) == variant]
    if not files:
        return None
    runs = [read_jsonl(f) for f in files]
    qids = [r["qid"] for r in runs[0]]
    f1 = np.mean([[r["f1"] for r in run] for run in runs], axis=0)
    return qids, f1


def bootstrap_markdown(out: Path, n: int, variants: list[str]) -> str:
    lines = ["| variant | F1 | 95% CI | F1 - rag | 95% CI of diff |", "|---|---|---|---|---|"]
    ref = _per_question(out, "rag")
    for v in variants:
        pq = _per_question(out, v)
        if pq is None:
            continue
        qids, f1 = pq
        lo, hi = bootstrap_ci(f1, n)
        cell = "| — | — |"
        if ref is not None and v != "rag" and ref[0] == qids:
            d = paired_bootstrap(ref[1], f1, n)
            cell = f"| {d['diff']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] |"
        lines.append(f"| {v} | {f1.mean():.4f} | [{lo:.4f}, {hi:.4f}] " + cell)
    lines += ["", "Fine-tuned variants use per-question F1 averaged over seeds. Paired bootstrap resamples questions "
              f"({n} resamples); a CI that excludes 0 means the difference is unlikely to be resampling noise."]
    return "\n".join(lines)


def _bucket(r: dict[str, Any]) -> str:
    if r["is_impossible"]:
        return "unanswerable: abstained (correct)" if r["abstained"] else "unanswerable: answered (hallucination)"
    if r["f1"] >= 0.5:
        return "answerable: correct (F1>=0.5)"
    if r.get("context") is not None and not r["retrieval_hit"]:
        return "answerable: retrieval miss"
    if r["abstained"]:
        return "answerable: false abstain"
    return "answerable: wrong span"


def error_analysis_markdown(out: Path, variants: list[str], n_examples: int) -> str:
    lines = ["## Outcome buckets (share of test questions)", ""]
    runs: dict[str, list[dict[str, Any]]] = {}
    for v in variants:
        files = sorted(f for f in (out / "predictions").glob(f"{v}*.jsonl") if _variant_of(f.stem) == v)
        if files:
            runs[v] = read_jsonl(files[0])  # first seed for fine-tuned variants
    buckets = sorted({_bucket(r) for recs in runs.values() for r in recs})
    lines += ["| bucket | " + " | ".join(runs) + " |", "|" + "---|" * (len(runs) + 1)]
    for b in buckets:
        cells = [f"{np.mean([_bucket(r) == b for r in recs]):.3f}" for recs in runs.values()]
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines += ["", "## F1 by question type", ""]
    types = Counter(r["qtype"] for r in next(iter(runs.values()), []))
    lines += ["| type | n | " + " | ".join(runs) + " |", "|" + "---|" * (len(runs) + 2)]
    for t, cnt in types.most_common():
        cells = [f"{np.mean([r['f1'] for r in recs if r['qtype'] == t]):.3f}" for recs in runs.values()]
        lines.append(f"| {t} | {cnt} | " + " | ".join(cells) + " |")
    focus = "rag_ft" if "rag_ft" in runs else ("rag" if "rag" in runs else next(iter(runs), None))
    if focus:
        lines += ["", f"## Failure examples ({focus})", ""]
        for b in buckets:
            if "correct" in b:
                continue
            ex = [r for r in runs[focus] if _bucket(r) == b][:n_examples]
            if not ex:
                continue
            lines += [f"### {b}", ""]
            for r in ex:
                gold = "; ".join(r["golds"]) or "(no answer)"
                lines.append(f"- Q: {r['question']} | gold: {gold} | reply: {r['reply'].strip()[:120]!r}")
            lines.append("")
    return "\n".join(lines)


def write_reports(cfg: ExperimentConfig, out: Path) -> list[dict[str, Any]]:
    rows = collect_results(out)
    if not rows:
        return rows
    (out / "results.jsonl").write_text("", encoding="utf-8")
    for r in rows:
        append_jsonl(r, out / "results.jsonl")
    (out / "results.md").write_text(results_markdown(rows) + "\n", encoding="utf-8")
    variants = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in rows)]
    (out / "bootstrap.md").write_text(bootstrap_markdown(out, cfg.n_bootstrap, variants) + "\n", encoding="utf-8")
    (out / "error_analysis.md").write_text(error_analysis_markdown(out, variants, cfg.n_examples) + "\n",
                                           encoding="utf-8")
    return rows


# ---------------------------------------------------------------- driver


def run_experiment(cfg: ExperimentConfig, backends: Backends | None = None) -> list[dict[str, Any]]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    device = resolve_device(cfg.device)
    backends = backends or hf_backends(cfg, device)
    log.info("device %s, output %s", device, out)

    ds = build_dataset(cfg.data)
    write_json(ds.summary, out / "data_summary.json")
    choice = run_retrieval(cfg, ds, backends, out)
    log.info("chosen retrieval: %s", choice)
    ctx = json.loads((out / "contexts.json").read_text(encoding="utf-8"))
    r = cfg.retrieval
    chunks = make_chunks(ds.dev_articles, ctx["chunker"], r.window_words, r.window_overlap)
    support = SupportIndex(chunks)
    contexts = ctx["contexts"]
    pred_dir = out / "predictions"

    for variant in [v for v in BASE_VARIANTS if v in cfg.gen.variants]:
        path = pred_dir / f"{variant}.jsonl"
        if path.exists():
            continue
        gen = backends.generator(None, variant == "rag_nf4")
        recs, ms = generate_variant(variant, ds.test_questions, gen, chunks, contexts, support)
        gen.close()
        free_memory()
        _write_predictions(recs, ms, path)
        log.info("%s: %s", variant, {k: round(v, 4) for k, v in summarize(recs).items()})

    for mode in cfg.ft.modes:
        variant = FT_VARIANT[mode]
        for seed in cfg.seeds:
            path = pred_dir / f"{variant}_s{seed}.jsonl"
            if path.exists():
                continue
            adapter = out / "adapters" / f"{mode}_s{seed}"
            if not (adapter / "adapter_config.json").exists():
                examples = sft_examples(mode, cfg, ds, ctx["chunker"], seed)
                info = backends.trainer(examples, seed, adapter)
                write_json(info, out / "ft_logs" / f"{mode}_s{seed}.json")
                free_memory()
            gen = backends.generator(str(adapter), False)
            recs, ms = generate_variant(variant, ds.test_questions, gen, chunks, contexts, support)
            gen.close()
            free_memory()
            _write_predictions(recs, ms, path)
            log.info("%s seed %d: f1 %.4f", variant, seed, summarize(recs)["f1"])

    rows = write_reports(cfg, out)
    log.info("done: %d result rows in %s", len(rows), out / "results.md")
    return rows


__all__ = ["ABSTAIN", "run_experiment", "score_record", "sft_examples", "write_reports"]
