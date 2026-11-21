| model | device | batch | ms/text | batch p50 ms | batch p95 ms | texts/s |
|---|---|---|---|---|---|---|
| tfidf_lr | cpu | 1 | 9.425 | 9.35 | 10.12 | 106 |
| tfidf_lr | cpu | 32 | 0.406 | 12.68 | 14.20 | 2463 |
| distilbert | cpu | 1 | 29.137 | 28.51 | 33.75 | 34 |
| distilbert | cpu | 32 | 12.405 | 343.82 | 637.39 | 81 |
| distilbert | cuda | 1 | 5.618 | 5.54 | 6.11 | 178 |
| distilbert | cuda | 32 | 0.344 | 10.48 | 14.69 | 2904 |

CPU logical cores visible: [4]. Batch-1 p95 is per-request latency.