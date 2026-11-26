# Training results (held-out test, mean ± std over 3 seeds)

| model | pr_auc | roc_auc | brier | ece | f1 | threshold | fit_s |
|---|---|---|---|---|---|---|---|
| logreg | 0.4977 ± 0.0000 | 0.7387 ± 0.0000 | 0.1435 ± 0.0000 | 0.0147 ± 0.0000 | 0.5115 ± 0.0027 | 0.2696 ± 0.0138 | 1.3502 ± 0.1459 |
| lgbm | 0.5551 ± 0.0025 | 0.7750 ± 0.0003 | 0.1355 ± 0.0001 | 0.0089 ± 0.0025 | 0.5399 ± 0.0004 | 0.2783 ± 0.0075 | 35.9218 ± 4.3369 |

Gate: candidate PR-AUC margin 0.0574 must exceed 1.0 × its seed std, and its ECE must be ≤ 0.03. Promoted: **True**. Served model: `lgbm-25ce825-202611221646` (threshold 0.2837, chosen on out-of-fold train predictions).
