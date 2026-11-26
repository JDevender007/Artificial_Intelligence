# Cost estimate

Derived from the measured Locust run in `reports/locust/run_stats.csv`, not from a cloud bill. The
only cost input this repository supplies is **CPU time per request**; the price per CPU-hour must be
read from the provider's current page, so no dollar figure is asserted here.

## What was measured

| quantity | value | source |
|---|---|---|
| aggregated throughput | 27.33 req/s | `run_stats.csv`, 20 users, 30 s |
| single-row `/predict` throughput | 24.54 req/s | `run_stats.csv` |
| batch-of-50 throughput | 2.48 req/s | `run_stats.csv` |
| p95 latency | 740 ms | `run_stats.csv` |
| failures | 0 / 715 | `run_stats.csv` |

Caveat: Locust ran on the same 12-core laptop as the single-process uvicorn worker, so the
throughput above is a lower bound. Container CPU and memory were not captured (Docker Desktop was
stopped); the CI `image` job records `docker stats` in the `p11-ci-reports` artifact.

## CPU time per 1,000,000 predictions

```
seconds_per_million = 1_000_000 / throughput_rps
```

| workload | throughput | seconds per 1M | core-hours per 1M |
|---|---|---|---|
| single-row `/predict` only | 24.54 req/s | 40,750 s | 11.32 |
| measured mix (90% single, 9% batch-of-50, 1% `/model-info`) | 27.33 req/s | 36,589 s | 10.16 |

So the cost of one million predictions is `10.16 × price_per_core_hour` in the currency of the
provider's price list. At a 2 vCPU instance, that is at most ~5.1 wall-clock hours **if** the two
workers scale linearly — which is not verified here, because feature engineering is memory-bound
pandas work per process.

## Where the time goes

`latency_profile.md` splits the single-row path: 20.36 ms median, of which 8.59 ms is pandas feature
engineering and 10.48 ms is the 5-fold calibrated classifier. The transformer is ~42% of the request.
A cheaper deployment would precompute the six-month aggregates in a feature store instead of
per request; that is the single largest lever and it is not implemented.

## Not costed

- **Memory:** unmeasured, so no instance-size recommendation is made.
- **Storage and egress:** the image is `python:3.13-slim` + LightGBM + pandas; the model is 6.8 MB
  (`models/model.joblib`). Neither dominates the bill at any realistic volume.
- **Cold start:** unmeasured. The lifespan hook loads `model.joblib` once at startup.
- **Scale:** there is no autoscaling or queueing analysis; 20 concurrent users is the tested ceiling.
