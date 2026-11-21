# P4 career assets

**Repo folder:** `p4-intent-classification` (suggested standalone name: `banking77-intent-tfidf-vs-distilbert`)

**Resume bullet**
- Built a 77-class banking intent classifier. A tuned TF-IDF + logistic regression baseline reached 0.913 test macro-F1 at 2.3 ms per message on CPU. [ ] DistilBERT fine-tune: X macro-F1 (+Y, 95% bootstrap CI [a, b]) at Z ms on a T4. The project also ships a model card, error analysis by class and by message length, and CI-tested code.

**LinkedIn post (draft)**
> Is a transformer worth it for intent classification? I compared a tuned TF-IDF + logistic regression baseline with a fine-tuned DistilBERT on Banking77 (77 intents, CC BY 4.0).
>
> - The baseline already reaches 0.913 macro-F1 and answers in about 2 ms on a CPU.
> - [ ] DistilBERT: X macro-F1, Y ms on CPU / Z ms on GPU.
> - Most remaining errors are between near-duplicate intents (`top_up_failed` vs `top_up_reverted`, `verify_my_identity` vs `why_verify_identity`).
>
> Code, tests and the model card: [repo link]
