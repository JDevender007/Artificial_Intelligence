"""SQuAD 2.0 loading, article-level split and sampling.

Each article's paragraphs are joined with a blank line into one article text, so every answer becomes an
article-level character span. Chunkers then cut the article text however they like, and a chunk "supports" a
question when it contains a whole gold answer span (see :mod:`ragqa.chunking`).

SQuAD 2.0 (Rajpurkar et al., 2018) is CC BY-SA 4.0. Its train and dev sets use disjoint Wikipedia articles.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ragqa.config import DataConfig
from ragqa.utils import get_logger

PARA_SEP = "\n\n"
log = get_logger()


@dataclass(frozen=True)
class Article:
    title: str
    text: str
    para_offsets: tuple[int, ...]


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    article: str
    para_index: int
    answers: tuple[str, ...]  # empty for unanswerable questions
    starts: tuple[int, ...]  # article-level character offsets, aligned with ``answers``
    is_impossible: bool


@dataclass
class Dataset:
    dev_articles: list[Article]
    train_articles: list[Article]
    val_questions: list[Question]
    test_questions: list[Question]
    train_questions: list[Question]  # the QLoRA training sample
    summary: dict[str, Any]


def parse_squad(obj: dict[str, Any]) -> tuple[list[Article], list[Question], int]:
    """Articles, questions and the number of answer spans dropped because the text did not match the offset."""
    articles, questions, dropped = [], [], 0
    for art in obj["data"]:
        title = art["title"].replace("_", " ")
        paras = [p["context"] for p in art["paragraphs"]]
        offsets, pos = [], 0
        for p in paras:
            offsets.append(pos)
            pos += len(p) + len(PARA_SEP)
        text = PARA_SEP.join(paras)
        articles.append(Article(title, text, tuple(offsets)))
        for pi, p in enumerate(art["paragraphs"]):
            for qa in p["qas"]:
                pairs: dict[tuple[str, int], None] = {}
                for a in qa.get("answers", []):
                    s = offsets[pi] + int(a["answer_start"])
                    if text[s:s + len(a["text"])] != a["text"]:
                        dropped += 1
                        continue
                    pairs[(a["text"], s)] = None
                imp = bool(qa.get("is_impossible", False))
                if not imp and not pairs:
                    continue  # answerable but no usable span: cannot be scored
                questions.append(Question(qa["id"], qa["question"].strip(), title, pi,
                                          tuple(a for a, _ in pairs), tuple(s for _, s in pairs), imp))
    return articles, questions, dropped


def download(url: str, path: Path, retries: int = 5) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d)", url, attempt)
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(path)
            return path
        except OSError as e:
            log.warning("download failed: %s", e)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {url}")


# ---------------------------------------------------------------- synthetic data for tests

_SYL = ["zor", "vi", "plim", "ton", "kas", "ret", "mun", "dra", "lo", "fen", "quil", "bar", "sen", "tos", "nal", "gri"]


def _name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        n = "".join(rng.choice(_SYL, size=int(rng.integers(2, 4)))).capitalize()
        if n not in used:
            used.add(n)
            return n


def synthetic_squad(n_articles: int, seed: int) -> dict[str, Any]:
    """Small SQuAD-format corpus of invented countries: 3 paragraphs and 6 questions (1 unanswerable) each."""
    rng = np.random.default_rng(seed)
    used: set[str] = set()
    data = []
    for i in range(n_articles):
        c, cap, riv, fnd, exp, crop = (_name(rng, used) for _ in range(6))
        year = str(int(rng.integers(1100, 1990)))
        paras = [
            f"{c} is a small country in the north. Its capital is {cap}, a city on the {riv} river.",
            f"{c} was founded in {year} by {fnd}. The founding is celebrated every spring with music.",
            f"The main export of {c} is {exp}. Farmers in the valleys grow {crop} and keep goats.",
        ]
        qas = [(0, f"What is the capital of {c}?", cap), (0, f"Which river runs through {cap}?", riv),
               (1, f"In what year was {c} founded?", year), (1, f"Who founded {c}?", fnd),
               (2, f"What is the main export of {c}?", exp), (2, f"What is the national bird of {c}?", None)]
        paragraphs = []
        for pi, ctx in enumerate(paras):
            items = []
            for k, (qpi, q, a) in enumerate(qas):
                if qpi != pi:
                    continue
                ans = [] if a is None else [{"text": a, "answer_start": ctx.index(a)}]
                items.append({"id": f"s{seed}_{i}_{k}", "question": q, "answers": ans, "is_impossible": a is None})
            paragraphs.append({"context": ctx, "qas": items})
        data.append({"title": f"{c}_{seed}", "paragraphs": paragraphs})
    return {"version": "v2.0", "data": data}


# ---------------------------------------------------------------- dataset assembly


def _load_split(cfg: DataConfig, split: str) -> dict[str, Any]:
    raw = Path(cfg.raw_dir)
    if cfg.synthetic:
        p = raw / f"synthetic-{split}.json"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            n = cfg.synthetic_articles * (2 if split == "train" else 1)
            p.write_text(json.dumps(synthetic_squad(n, seed=1 if split == "train" else 0)), encoding="utf-8")
    else:
        url = cfg.train_url if split == "train" else cfg.dev_url
        p = download(url, raw / Path(url).name)
    return json.loads(p.read_text(encoding="utf-8"))


def _sample(qs: list[Question], n: int, rng: np.random.Generator) -> list[Question]:
    if n >= len(qs):
        return list(qs)
    idx = np.sort(rng.choice(len(qs), size=n, replace=False))
    return [qs[i] for i in idx]


def _qstats(qs: list[Question]) -> dict[str, Any]:
    ans_words = [len(q.answers[0].split()) for q in qs if q.answers]
    return {"questions": len(qs),
            "unanswerable_frac": float(np.mean([q.is_impossible for q in qs])) if qs else float("nan"),
            "answer_words_mean": float(np.mean(ans_words)) if ans_words else float("nan")}


def _stats(articles: list[Article], qs: list[Question]) -> dict[str, Any]:
    n_par = sum(len(a.para_offsets) for a in articles)
    words = [len(p.split()) for a in articles for p in a.text.split(PARA_SEP)]
    return {"articles": len(articles), "paragraphs": n_par, **_qstats(qs),
            "paragraph_words_mean": float(np.mean(words)) if words else float("nan"),
            "paragraph_words_p95": float(np.percentile(words, 95)) if words else float("nan")}


def build_dataset(cfg: DataConfig) -> Dataset:
    dev_a, dev_q, dev_drop = parse_squad(_load_split(cfg, "dev"))
    tr_a, tr_q, tr_drop = parse_squad(_load_split(cfg, "train"))
    overlap = {a.title for a in dev_a} & {a.title for a in tr_a}
    if overlap:
        raise ValueError(f"train and dev share articles: {sorted(overlap)[:5]}")
    rng = np.random.default_rng(cfg.split_seed)
    titles = sorted(a.title for a in dev_a)
    if cfg.n_val_articles >= len(titles):
        raise ValueError("n_val_articles must leave at least one test article")
    val_titles = {titles[i] for i in rng.permutation(len(titles))[: cfg.n_val_articles]}
    val_pool = [q for q in dev_q if q.article in val_titles]
    test_pool = [q for q in dev_q if q.article not in val_titles]
    val_q = _sample(val_pool, cfg.n_val_questions, rng)
    test_q = _sample(test_pool, cfg.n_test_questions, rng)
    train_q = _sample(tr_q, cfg.n_train_examples, rng)
    summary = {
        "source": "synthetic" if cfg.synthetic else "SQuAD 2.0 (CC BY-SA 4.0)",
        "dev_corpus": _stats(dev_a, dev_q), "train_corpus": _stats(tr_a, tr_q),
        "dropped_spans": {"dev": dev_drop, "train": tr_drop},
        "val": {"articles": len(val_titles), "pool": len(val_pool), **_qstats(val_q)},
        "test": {"articles": len(titles) - len(val_titles), "pool": len(test_pool), **_qstats(test_q)},
        "train_sample": _qstats(train_q),
    }
    return Dataset(dev_a, tr_a, val_q, test_q, train_q, summary)
