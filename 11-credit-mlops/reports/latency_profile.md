| step | rows | median ms | p95 ms |
|---|---|---|---|
| full pipeline | 1 | 20.36 | 26.27 |
| feature engineering (pandas) | 1 | 8.59 | 11.73 |
| classifier | 1 | 10.48 | 12.77 |
| full pipeline | 50 | 23.02 | 25.89 |
| feature engineering (pandas) | 50 | 8.59 | 11.56 |
| classifier | 50 | 13.44 | 16.32 |
| full pipeline | 1000 | 71.92 | 78.23 |
| feature engineering (pandas) | 1000 | 10.65 | 12.96 |
| classifier | 1000 | 62.57 | 68.05 |

Classifier = 5 calibrated LightGBM folds, averaged; n_jobs=4 per booster.
