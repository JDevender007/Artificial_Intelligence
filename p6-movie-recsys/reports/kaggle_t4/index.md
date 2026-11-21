# Retrieval index (two_tower seed 0, 3883 items, dim 64, 1188 test users, k=20)

recall_vs_exact = overlap of raw top-20 with exact search (before seen-item filtering); latency = one query at a time on this machine's CPU.

| index | recall_vs_exact | p50_ms | p95_ms | build_s |
|---|---|---|---|---|
| numpy | 1.0000 | 0.1218 | 0.1508 | 0.0000 |
| flat | 1.0000 | 0.0526 | 0.0702 | 0.0003 |
| hnsw(ef=16) | 0.9483 | 0.0212 | 0.0307 | 0.3698 |
| hnsw(ef=32) | 0.9899 | 0.0273 | 0.0404 | 0.3709 |
| hnsw(ef=64) | 0.9987 | 0.0376 | 0.0494 | 0.3678 |
| hnsw(ef=128) | 0.9998 | 0.0576 | 0.0728 | 0.3720 |
