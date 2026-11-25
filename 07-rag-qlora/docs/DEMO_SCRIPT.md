# P7 60-second demo script

1. **(0-8 s)** "An open-book QA system over SQuAD 2.0. It answers from retrieved passages, cites the passage, and says 'I don't know' when the corpus doesn't contain the answer."

2. **(8-22 s)** Run `python scripts\ask.py "Who wrote the novel Dune?" --config configs\squad.yaml` and point at the trace: retrieved passage ids, the 5 passages given to the model, the answer, and the cited id. "Retrieval is hybrid — BM25 plus a BGE dense index fused with reciprocal rank, then reranked — and the citation is checked, not decorative."

3. **(22-38 s)** Open `reports/kaggle_t4/retrieval.md`. "This is the ablation: two chunkers, four methods, k of 1, 5 and 20, on validation then test. The setup was chosen on validation recall@5, never on test scores, so the test column is the first time these numbers are seen."

4. **(38-52 s)** Open `reports/kaggle_t4/results.md`. "Four arms on the same questions: base model, prompt-only, RAG, and a 4-bit QLoRA fine-tune. F1 is reported on the full set plus answerable-only, next to no-answer accuracy and the false-abstain rate — because a model that refuses everything looks perfect on abstention."

5. **(52-60 s)** Open `reports/kaggle_t4/error_analysis.md`. "Failures are bucketed into retrieval misses, wrong answers with the right passage, and false abstains, so I can say which half is broken." End on the green CI badge.
