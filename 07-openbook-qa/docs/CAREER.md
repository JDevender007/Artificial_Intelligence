# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/07-openbook-qa` (monorepo). Standalone name if split out: `openbook-qa`

**Resume bullet** (fill the bracketed numbers from `reports/kaggle_t4/results.md` once the run finishes — do not guess)
- Built an open-book QA system on SQuAD 2.0 (CC BY-SA 4.0) with a leakage-free article-level split, a 2×4×3 retrieval ablation (paragraph vs 100-word windows, BM25 / dense / hybrid RRF / hybrid + BGE reranker) and four generation arms: base, prompt-only, RAG and a QLoRA fine-tune of Qwen2.5-1.5B. RAG moved full-test F1 from [X] to [Y] and answerable-only F1 from [X] to [Y], with citations that actually contain the answer in [Z]% of cases; the fine-tune traded [A] points of faithfulness for [B] points of recall on retrieved passages.

**LinkedIn post (draft)**
> Most RAG demos show one answer and stop. The interesting question is which half is broken — the retriever or the reader.
>
> So I built the boring version: SQuAD 2.0, an article-level split so no test question shares a Wikipedia article with training data, and four generation arms scored on the same 1,000 questions:
> - retrieval ablation first: paragraph vs 100-word windows × BM25 / dense / hybrid / hybrid + reranker, chosen on validation recall@5 only
> - base vs prompted vs RAG vs a 4-bit QLoRA fine-tune of Qwen2.5-1.5B
> - abstention measured as a rate, because "I don't know" is the right answer to half of SQuAD 2.0
>
> What I expect from the numbers and will confirm, not assume: retrieval fixes the factual questions, the fine-tune fixes the citation format, and neither fixes a passage the retriever never returned. When the run lands, the tables go in the repo next to the raw predictions — [repo link]

**Do not publish this draft until `reports/` exists.** The bullet has placeholders for a reason.
