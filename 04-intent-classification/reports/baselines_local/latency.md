| model | device | batch | ms/text | batch p50 ms | batch p95 ms | texts/s |
|---|---|---|---|---|---|---|
| tfidf_lr | cpu | 1 | 2.293 | 2.16 | 3.03 | 436 |
| tfidf_lr | cpu | 32 | 0.240 | 7.37 | 9.12 | 4159 |

CPU logical cores visible: [16]. Batch-1 p95 is per-request latency.