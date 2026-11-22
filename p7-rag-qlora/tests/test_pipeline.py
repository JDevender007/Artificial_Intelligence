from pathlib import Path

import pytest

from ragqa.config import load_config
from ragqa.data import build_dataset
from ragqa.experiment import run_experiment, score_record, sft_examples
from ragqa.generate import fake_backends
from ragqa.prompts import ABSTAIN

CFG = Path(__file__).resolve().parents[1] / "configs" / "test.yaml"


@pytest.fixture()
def cfg(tmp_path):
    return load_config(CFG, [f"output_dir={tmp_path / 'out'}", f"data.raw_dir={tmp_path / 'raw'}"])


def test_split_has_no_leakage(cfg):
    ds = build_dataset(cfg.data)
    assert not {a.title for a in ds.train_articles} & {a.title for a in ds.dev_articles}
    assert not {q.article for q in ds.val_questions} & {q.article for q in ds.test_questions}
    assert {q.article for q in ds.train_questions} <= {a.title for a in ds.train_articles}
    assert any(q.is_impossible for q in ds.test_questions)


def test_config_validation():
    with pytest.raises(ValueError):
        load_config(CFG, ["retrieval.k_context=3"])  # not in ks
    with pytest.raises(KeyError):
        load_config(CFG, ["gen.nope=1"])


def test_sft_examples_rag_targets(cfg):
    ds = build_dataset(cfg.data)
    ex = sft_examples("rag", cfg, ds, "paragraph", seed=0)
    assert len(ex) == len(ds.train_questions)
    n_cited = 0
    for e, q in zip(ex, ds.train_questions, strict=True):
        if q.is_impossible or e["target"] == ABSTAIN:
            continue
        ans, cite = e["target"].rsplit(" [", 1)
        n = int(cite.rstrip("]"))
        passage = e["messages"][1]["content"].split("\n")[n]  # line 0 is "Passages:"
        assert passage.startswith(f"[{n}] ") and ans in passage
        n_cited += 1
    assert n_cited > 0
    closed = sft_examples("closed", cfg, ds, "paragraph", seed=0)
    assert all(len(e["messages"]) == 2 for e in closed)


def test_score_record_fields(cfg):
    ds = build_dataset(cfg.data)
    q = next(q for q in ds.test_questions if not q.is_impossible)
    rec = score_record(q, f"{q.answers[0]} [1]", None, set())
    assert rec["em"] == 1.0 and rec["context"] is None and rec["citations"] == [1]


def test_full_pipeline_with_fakes_and_resume(cfg):
    rows = run_experiment(cfg, fake_backends())
    out = Path(cfg.output_dir)
    assert {r["variant"] for r in rows} == {"base", "prompted", "rag", "rag_nf4", "ft_closed", "rag_ft"}
    assert sum(r["variant"] == "rag_ft" for r in rows) == len(cfg.seeds)
    for f in ("data_summary.json", "retrieval.md", "retrieval_choice.json", "contexts.json", "results.md",
              "bootstrap.md", "error_analysis.md", "config.yaml"):
        assert (out / f).exists(), f
    rag = next(r for r in rows if r["variant"] == "rag")
    assert 0 <= rag["context_recall"] <= 1 and rag["f1"] > 0  # the rule generator copies words from passages
    closed = next(r for r in rows if r["variant"] == "prompted")
    assert closed["abstain_rate"] == 1.0  # no passages -> the fake abstains
    # resume: deleting one prediction file regenerates only that file
    stamp = (out / "predictions" / "rag.jsonl").stat().st_mtime_ns
    (out / "predictions" / "base.jsonl").unlink()
    rows2 = run_experiment(cfg, fake_backends())
    assert (out / "predictions" / "rag.jsonl").stat().st_mtime_ns == stamp
    assert len(rows2) == len(rows)
