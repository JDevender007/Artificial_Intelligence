# Open-book QA on SQuAD 2.0: retrieval ablation + QLoRA reader

![CI](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p7-ci.yml/badge.svg)

Open-book question answering that answers only from retrieved passages, cites the passage it used, and abstains when the corpus does not contain the answer. The point is not "a RAG demo": it is a **controlled four-way comparison** — base model, prompt-only, RAG, and a QLoRA fine-tune — on the same held-out questions, scored with the same metrics.

> **Status:** code, configs, tests and CI are complete (15 tests, ruff clean). The full run (`configs/squad.yaml`) has **not finished yet**, so this README has no results table. Nothing here is estimated: the numbers appear only from `reports/` after a run.

## Problem and success criteria

| | |
|---|---|
| Task | Extractive QA over a passage set, plus abstention on unanswerable questions (SQuAD 2.0 format) |
| Primary metric | **F1** on the full test set (answerable + unanswerable), and **EM** |
| Secondary | answerable-only F1, no-answer accuracy, false-abstain rate, citation rate, context recall, faithfulness, ms/question |
| Baseline to beat | the same model **without** retrieval (`base`), and prompt-only (`prompted`) |
| Comparison of interest | RAG vs QLoRA fine-tune (`ft_closed` = closed-book fine-tune, `rag_ft` = fine-tune trained to answer from retrieved passages) |
| Done when | 3 seeds per trained variant, bootstrap CI on the F1 difference vs `rag`, a retrieval ablation (2 chunkers × 4 methods × k ∈ {1,5,20}), and an error analysis by outcome bucket |

Why F1 on the *whole* test set: SQuAD 2.0 mixes answerable and unanswerable questions, so a model that always answers looks fine on F1 alone and a model that always abstains looks fine on no-answer accuracy. Reporting both, plus the false-abstain rate, makes the trade-off visible.

## Data

[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar, Jia & Liang, 2018). **License: CC BY-SA 4.0.** 150k questions, 50k of them unanswerable, written against ~500 Wikipedia articles.

Downloaded by `src/ragqa/data.py` straight from `rajpurkar.github.io` into `data/raw/` (with retries and a resume-safe temp file). No `datasets` dependency.

**Split — no leakage by construction:**

1. SQuAD's own **train** and **dev** article sets are disjoint, so fine-tuning data never shares an article with the evaluation questions.
2. The dev articles are split again, by article, into a **validation** set (used only to choose the chunker/method/k) and the **test** set.
3. `n_train_examples=3000` are sampled from SQuAD train only.
4. The retrieval setup is chosen on validation recall@k, never on test scores.

## Approach

```mermaid
flowchart LR
    A[SQuAD 2.0<br/>train + dev] --> B[split by article]
    B --> V[val articles<br/>choose retrieval setup]
    B --> T[test articles<br/>scored once]
    B --> F[train articles<br/>3000 SFT examples]
    T --> C{chunking<br/>paragraph | window 100/25}
    C --> R1[BM25]
    C --> R2[dense BGE-small]
    R1 --> H[hybrid RRF k=60]
    R2 --> H
    H --> RR[rerank BGE-reranker<br/>top-20 -> top-5]
    RR --> G{generator: Qwen2.5-1.5B-Instruct}
    G --> P1[base]
    G --> P2[prompted]
    G --> P3[rag / rag_nf4]
    F --> L[QLoRA r=16 alpha=32<br/>2 modes x 3 seeds]
    L --> P4[ft_closed / rag_ft]
    P1 --> M[EM, F1, no-ans acc,<br/>context recall, faithfulness,<br/>citation checks, ms/q]
    P2 --> M
    P3 --> M
    P4 --> M
```

Design choices, one recommendation each:

- **Retrieve 5 passages, not 20** — a 1.5B model's 32-token answer degrades with distractor count faster than recall@20 helps; 5 is chosen by validation recall@5.
- **Hybrid + rerank as the default**, not dense-only — BM25 wins on rare entities (species, numbers) and the reranker fixes most of dense's ordering errors.
- **Abstain by prompt + a fixed rule, not by a trained classifier** — one fewer model to fail silently, and the false-abstain rate makes the cost visible.
- **QLoRA r=16 on all projections** — on a T4 with 1.5B parameters, full fine-tuning is impossible and r=8 underfits the citation behaviour.
- **`p_drop_gold=0.2` in the RAG fine-tune** — 20% of answerable training examples are given without their gold passage, so the model learns to abstain instead of hallucinating when retrieval misses.

## Repository layout

```
07-openbook-qa/
├── configs/            squad.yaml (full), smoke.yaml (GPU, minutes), test.yaml (synthetic, no torch)
├── src/ragqa/          bm25 chunking config data retrieve generate hf metrics prompts experiment explain utils
├── scripts/            run_experiment.py  ask.py  make_notebook.py
├── tests/              test_retrieval, test_text_metrics, test_pipeline (fake backends, no downloads)
├── notebooks/          kaggle_run.ipynb  (thin wrapper: clone -> install -> smoke -> full)
├── pyproject.toml  requirements.txt  requirements-gpu.txt  requirements-dev.txt  requirements-ci.txt
└── README.md
```

`src/ragqa/hf.py` is the **only** module that imports torch/transformers/peft/bitsandbytes, and it imports them lazily — which is why the 15 tests run in CI with no GPU stack installed.

## Setup and run

```powershell
cd "D:\Projects\Artificial Intelligence\07-openbook-qa"
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt -r requirements.txt
pip install -e . --no-deps

pytest -q                      # 15 tests, fake backends, no download
ruff check .

# Ask a single question from the CLI (needs the GPU extras)
pip install -r requirements-gpu.txt
python scripts\ask.py "Who wrote the novel Dune?" --config configs\squad.yaml
```

Full experiment (Kaggle T4 or Colab; ~GPU hours, resumable — finished stages are skipped on re-run):

```powershell
python scripts\run_experiment.py --config configs\squad.yaml
```

Fast plumbing check on a GPU (0.5B model, 40 val questions, 4 QLoRA steps):

```powershell
python scripts\run_experiment.py --config configs\smoke.yaml
```

Paper-mill sanity check with no downloads at all (synthetic SQuAD-format files, fake backends):

```powershell
python scripts\run_experiment.py --config configs\test.yaml
```

`torch` is **not** pinned: the CUDA build comes from the Kaggle/Colab image. After a successful run, freeze what you actually got with `pip freeze > requirements-lock.txt`.

## Results

**Not run yet.** A completed run writes, under `outputs/squad/`:

| file | contents |
|---|---|
| `results.md` | mean ± std over seeds for every variant: EM, F1, answerable F1, no-answer accuracy, false abstain, context recall, faithfulness, citation rate, citation-supports-answer, citation-matches-gold, ms/question |
| `bootstrap.md` | per-question F1 with a 95% bootstrap CI, plus the paired CI of the difference against `rag` |
| `retrieval.md` | the ablation: val and test recall@{1,5,20}, MRR@10 and ms/query for 2 chunkers × 4 methods |
| `retrieval_choice.json` | which chunker/method won on validation, and why |
| `error_analysis.md` | outcome buckets (correct / wrong answer / wrongly answered unanswerable / false abstain) and failure examples |
| `predictions/*.jsonl` | every answer, its cited passage and the retrieved context, per variant and seed |

Those files then go to `reports/<run>/`, and the table below is filled from `results.md` — copied, never retyped.

## Metrics, defined

- **EM / F1** — SQuAD's official normalisation (lowercase, strip punctuation and articles) so numbers are comparable with the literature.
- **context_recall** — share of questions whose supporting passage appears in the 5 retrieved passages. Separates retrieval failure from generation failure.
- **faithfulness** — share of answers whose tokens are supported by the retrieved context (RAGAS-style, computed with overlap, not an LLM judge — no API cost, but stricter in what it can claim).
- **cite_rate** — share of answers that name a passage id; **cite_supports_answer** — that passage actually contains the answer span.
- **false_abstain** — share of *answerable* questions the model refused.

## Limitations

- One reader model (Qwen2.5-1.5B-Instruct) and one retriever family (BGE-small + BGE-reranker); the ranking of RAG vs fine-tune may not transfer to a 7B+ reader.
- Faithfulness by token overlap is a proxy; a model that paraphrases correctly can be scored as unfaithful.
- SQuAD passages are short Wikipedia paragraphs, so "chunking" here is easier than in a real corpus of PDFs.
- The fine-tune sees 3,000 examples for one epoch — enough to change behaviour, not to claim domain adaptation.
- Licence: SQuAD is CC BY-SA 4.0 (share-alike) and Qwen2.5 has its own licence; check both before redistributing a fine-tuned adapter.

## Ethics and responsible use

- QA over retrieved text can surface whichever passage ranks highest; the citation field is what makes an answer auditable, so it is measured, not decorative.
- Abstention is measured as a rate, because an assistant that refuses everything trivially scores 100% on unanswerable questions.
- The base model can still produce confident text that the context does not support; `faithfulness` exists to make that visible rather than hidden.

## Future work

- Swap BGE for a stronger retriever and report the delta on the same split.
- Add a real hallucination judge (an LLM) alongside the overlap metric and report both.
- Serve the retriever as a service with a vector index (FAISS/Qdrant) and measure end-to-end latency per question.
