# P4 career assets

**Repo folder:** `04-intent-classification` (suggested standalone name: `banking77-intent-tfidf-vs-distilbert`)

**Resume bullet**
- Built a 77-class banking intent classifier. A tuned TF-IDF + logistic regression baseline reached 0.913 test macro-F1 at 2.3 ms per message on CPU. A fine-tuned DistilBERT reached 0.921 ± 0.006 (3 seeds) and cut calibration error from 0.063 to 0.023 ECE, but a paired bootstrap showed the accuracy gain is within noise (95% CI [-0.008, +0.012]). On the same 4-core CPU it was 3× slower per request (29 vs 9 ms); on a T4 it ran at 5.6 ms. The project also ships a model card, error analysis by class and by message length, and CI-tested code.

**LinkedIn post (draft)**
> Is a transformer worth it for intent classification? I compared a tuned TF-IDF + logistic regression baseline with a fine-tuned DistilBERT on Banking77 (77 intents, CC BY 4.0).
>
> - The baseline already reaches 0.913 macro-F1 and answers in about 2 ms on a CPU.
> - DistilBERT: 0.921 ± 0.006 macro-F1, but the paired-bootstrap CI for the gain over the baseline includes zero. Its real win is calibration (ECE 0.023 vs 0.063).
> - Cost: 29 ms per request on a 4-core CPU (3× TF-IDF on the same machine), 5.6 ms on a T4.
> - The two models disagree on 209 test messages but differ by only 7 in net correct, so "just use the transformer" isn't automatically the right call.
> - Most remaining errors are between near-duplicate intents (`top_up_failed` vs `top_up_reverted`, `verify_my_identity` vs `why_verify_identity`).
>
> Code, tests and the model card: [repo link]
