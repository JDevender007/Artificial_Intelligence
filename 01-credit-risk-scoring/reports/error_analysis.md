# Error analysis: `xgboost_tuned_calibrated` (seed 0, threshold 0.298)

All numbers are computed on the held-out test set by `scripts/error_analysis.py`.


## By `recent_status`

| group              |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:-------------------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| delayed_last_month | 1324 |        0.498 |            0.518 |    0.730 |        0.926 | 0.678 |       0.576 |            0.801 |
| on_time_last_month | 4676 |        0.143 |            0.138 |    0.276 |        0.166 | 0.035 |       0.439 |            0.054 |


## By `months_delayed`

| group   |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:--------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| 0       | 4035 |        0.125 |            0.116 |    0.226 |        0.034 | 0.005 |       0.515 |            0.008 |
| 1-2     | 1221 |        0.306 |            0.334 |    0.526 |        0.706 | 0.357 |       0.466 |            0.464 |
| 3-6     |  744 |        0.603 |            0.610 |    0.769 |        0.982 | 0.929 |       0.617 |            0.961 |


## By `limit_quartile`

| group   |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:--------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| Q1_low  | 1530 |        0.303 |            0.328 |    0.608 |        0.676 | 0.268 |       0.523 |            0.392 |
| Q2      | 1547 |        0.241 |            0.242 |    0.584 |        0.566 | 0.141 |       0.561 |            0.243 |
| Q3      | 1465 |        0.166 |            0.171 |    0.462 |        0.399 | 0.067 |       0.542 |            0.122 |
| Q4_high | 1458 |        0.170 |            0.139 |    0.534 |        0.407 | 0.049 |       0.631 |            0.110 |


## By `age_band`

| group   |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:--------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| 30-39   | 2198 |        0.206 |            0.205 |    0.545 |        0.521 | 0.113 |       0.544 |            0.197 |
| 40-49   | 1341 |        0.233 |            0.220 |    0.559 |        0.519 | 0.122 |       0.562 |            0.215 |
| 50+     |  549 |        0.246 |            0.237 |    0.596 |        0.533 | 0.130 |       0.571 |            0.230 |
| <30     | 1912 |        0.223 |            0.238 |    0.566 |        0.590 | 0.144 |       0.541 |            0.244 |


## By `sex`

| group   |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:--------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| female  | 3598 |        0.213 |            0.217 |    0.566 |        0.539 | 0.118 |       0.553 |            0.208 |
| male    | 2402 |        0.234 |            0.228 |    0.551 |        0.551 | 0.140 |       0.545 |            0.236 |


## By `education`

| group       |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:------------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| graduate    | 2130 |        0.190 |            0.191 |    0.548 |        0.499 | 0.093 |       0.558 |            0.170 |
| high_school | 1014 |        0.257 |            0.246 |    0.534 |        0.521 | 0.158 |       0.533 |            0.251 |
| other       |   82 |        0.061 |            0.147 |    0.155 |        0.200 | 0.052 |       0.200 |            0.061 |
| university  | 2774 |        0.236 |            0.238 |    0.580 |        0.584 | 0.146 |       0.553 |            0.249 |


## By `marriage`

| group   |    n |   prevalence |   mean_pred_prob |   pr_auc |   tpr_recall |   fpr |   precision |   selection_rate |
|:--------|-----:|-------------:|-----------------:|---------:|-------------:|------:|------------:|-----------------:|
| married | 2767 |        0.241 |            0.221 |    0.593 |        0.531 | 0.124 |       0.576 |            0.223 |
| other   |   75 |        0.160 |            0.227 |    0.347 |        0.333 | 0.159 |       0.286 |            0.187 |
| single  | 3158 |        0.205 |            0.222 |    0.536 |        0.561 | 0.128 |       0.531 |            0.217 |


## Fairness gaps (max − min across groups)

Protected attributes were **excluded** from the model's inputs; they are used here only to audit.

|           |   tpr_gap |   fpr_gap |   selection_rate_gap |
|:----------|----------:|----------:|---------------------:|
| sex       |     0.012 |     0.022 |                0.028 |
| age_band  |     0.071 |     0.031 |                0.046 |
| education |     0.384 |     0.106 |                0.191 |
| marriage  |     0.228 |     0.034 |                0.036 |


## What do missed defaulters (FN) look like vs caught ones (TP)?

|                    |         TP |         FN |         FP |         TN |
|:-------------------|-----------:|-----------:|-----------:|-----------:|
| LIMIT_BAL          | 114819.945 | 164739.967 | 101959.459 | 187417.300 |
| PAY_1              |      1.511 |     -0.410 |      1.024 |     -0.395 |
| max_delay          |      2.281 |     -0.015 |      2.039 |     -0.063 |
| n_months_delayed   |      3.410 |      0.253 |      2.775 |      0.174 |
| util_1             |      0.540 |      0.405 |      0.557 |      0.389 |
| util_mean          |      0.525 |      0.346 |      0.516 |      0.334 |
| payratio_1         |      0.189 |      0.509 |      0.264 |      0.463 |
| payratio_mean      |      0.210 |      0.512 |      0.290 |      0.466 |
| pay_total_to_limit |      0.190 |      0.207 |      0.208 |      0.250 |
| count              |    722.000 |    605.000 |    592.000 |   4081.000 |


## Most confident false negatives (defaulted, scored lowest)

|        ID |   LIMIT_BAL |    AGE |   PAY_1 |   PAY_2 |   PAY_3 |   BILL_AMT1 |   PAY_AMT1 |     p |
|----------:|------------:|-------:|--------:|--------:|--------:|------------:|-----------:|------:|
|   983.000 |  500000.000 | 36.000 |  -2.000 |  -2.000 |  -2.000 |   45106.000 |  81690.000 | 0.003 |
| 28994.000 |  480000.000 | 39.000 |   0.000 |   0.000 |   0.000 |   98569.000 |  10620.000 | 0.019 |
| 16277.000 |  300000.000 | 31.000 |   0.000 |   0.000 |   0.000 |   34515.000 |  30000.000 | 0.026 |
| 22263.000 |  420000.000 | 31.000 |   0.000 |   0.000 |   0.000 |  233413.000 |  12000.000 | 0.028 |
| 22712.000 |  320000.000 | 35.000 |   0.000 |   0.000 |   0.000 |  157249.000 |   6000.000 | 0.033 |
|   971.000 |  300000.000 | 42.000 |  -1.000 |   0.000 |   0.000 |   11973.000 |  20979.000 | 0.033 |
|  8959.000 |  340000.000 | 44.000 |   0.000 |   0.000 |   0.000 |   83059.000 |  20000.000 | 0.034 |
|  5002.000 |  500000.000 | 45.000 |  -1.000 |  -1.000 |  -1.000 |    5307.000 |   6616.000 | 0.034 |
|  9796.000 |  250000.000 | 40.000 |   0.000 |   0.000 |   0.000 |  142477.000 |   7000.000 | 0.035 |
|  3792.000 |  120000.000 | 28.000 |  -1.000 |   0.000 |   0.000 |    8922.000 |  35000.000 | 0.036 |

## Most confident false positives (did not default, scored highest)

|        ID |   LIMIT_BAL |    AGE |   PAY_1 |   PAY_2 |   PAY_3 |   BILL_AMT1 |   PAY_AMT1 |     p |
|----------:|------------:|-------:|--------:|--------:|--------:|------------:|-----------:|------:|
| 28748.000 |   50000.000 | 28.000 |   3.000 |   2.000 |   2.000 |    2400.000 |      0.000 | 0.967 |
|  9672.000 |  170000.000 | 48.000 |   2.000 |   2.000 |   7.000 |    2400.000 |      0.000 | 0.908 |
| 26713.000 |   30000.000 | 26.000 |   2.000 |   2.000 |   2.000 |   14598.000 |   2000.000 | 0.881 |
| 14923.000 |   20000.000 | 25.000 |   2.000 |   2.000 |   2.000 |    8880.000 |   1500.000 | 0.868 |
| 22567.000 |  120000.000 | 30.000 |   2.000 |   2.000 |   2.000 |   69443.000 |   3200.000 | 0.835 |
|  1522.000 |   60000.000 | 40.000 |   3.000 |   3.000 |   5.000 |    9744.000 |   1500.000 | 0.831 |
| 10024.000 |   50000.000 | 39.000 |   3.000 |   2.000 |   2.000 |    5171.000 |      0.000 | 0.816 |
|  2102.000 |  140000.000 | 35.000 |   2.000 |   2.000 |   2.000 |   56528.000 |   2800.000 | 0.812 |
|   355.000 |   50000.000 | 58.000 |   2.000 |   2.000 |   2.000 |   15899.000 |   3400.000 | 0.804 |
|   215.000 |  470000.000 | 27.000 |   2.000 |   2.000 |   2.000 |  296573.000 |  13000.000 | 0.792 |

## Interpretation

- **The model is mostly a "recent delay" detector.** Clients who were late last month are flagged 80% of the time (recall 0.93, FPR 0.68); clients who paid on time are flagged 5% of the time (recall 0.17). Most missed defaulters (605 FN) look like good payers: average PAY_1 −0.41 vs 1.51 for caught defaulters.
- **Missed defaulters have high limits and high repayment ratios.** FN average limit is 164.7K vs 114.8K for TP, and they repaid about 51% of last bill vs 19%. The history gives no early signal for these sudden defaults; new features (e.g. income, other debts) would be needed.
- **False positives are chronic-but-paying clients.** The most confident FPs are 2–3 months behind every month yet did not default; the data cannot separate "always a bit late" from "about to default".
- **Fairness gaps are small for sex (TPR gap 0.012) and age (0.071)** even though those attributes were excluded. **Education and marriage show large gaps (TPR gap 0.38 and 0.23)**, but these come from tiny "other" groups (n = 82 and 75), so they are mostly noise and should not be over-read.
- **Low-limit clients (Q1) are flagged 39% of the time vs 11% for Q4,** tracking their higher base rate (0.30 vs 0.17). This is a known proxy for income and would need a policy review before any real use.
