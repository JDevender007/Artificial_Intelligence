# Drift drills (simulated scenarios on the held-out test set)

| scenario | drift alert | drifted cols | top PSI column | score PSI | default rate | mean P | PR-AUC | ECE |
|---|---|---|---|---|---|---|---|---|
| none | False | 0/19 | BILL_AMT4 (0.006) | 0.001 | 0.228 | 0.223 | 0.5647 | 0.0125 |
| limit_cut | False | 1/19 | LIMIT_BAL (0.518) | 0.043 | 0.228 | 0.231 | 0.5569 | 0.0160 |
| recession | False | 0/19 | PAY_1 (0.103) | 0.127 | 0.403 | 0.263 | 0.5772 | 0.1420 |
