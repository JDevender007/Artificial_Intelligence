# P7 interview Q&A

**1. Why build the retrieval ablation before the generation comparison?**
Because a reader that never sees the supporting passage cannot answer, and that failure is invisible in an aggregate F1. The ablation runs first on validation only — two chunkers, four retrieval methods, k of 1/5/20 — and picks the setup by recall@5. Then the generation arms are compared on test with that setup frozen, so a RAG gain is a reader/generator gain, not a lucky retriever.

**2. How do you know there is no train/test leakage?**
SQuAD 2.0's train and dev sets are built from disjoint Wikipedia articles, and I split the dev articles again into a validation set and a test set. Fine-tuning examples are sampled from SQuAD train articles only. So no question in the test set can be answered by memorising a paragraph that also appeared in training, and the retrieval index is built from the articles of the split the question belongs to.

**3. Why hybrid retrieval plus a reranker instead of dense-only?**
BM25 wins where exact tokens matter — names, numbers, rare entities — and dense retrieval wins on paraphrase. Reciprocal-rank fusion gets both candidates cheaply, and a cross-encoder reranker fixes the ordering of the top 20. It costs latency, which is why `ms/query` and `ms/question` are reported rather than hidden.

**4. Why QLoRA, and why is 20% of the training data given without its gold passage?**
On a free T4 a 1.5B model cannot be fully fine-tuned, so 4-bit NF4 with rank-16 adapters is the only realistic option. Dropping the gold passage from 20% of answerable training examples teaches the model to abstain when retrieval misses instead of inventing an answer — the failure mode that makes RAG demos look good and production systems dangerous.

**5. Your faithfulness metric is token overlap, not an LLM judge. What can and cannot you claim?**
It can claim that the answer's tokens appear in the retrieved context, which catches invented entities and numbers. It cannot claim semantic faithfulness: a correct paraphrase can score low, and a copy of a passage can score high while answering the wrong question. That is why it sits next to `cite_supports_answer` (the cited passage actually contains the answer span) and why adding an LLM judge is future work rather than a claim.
