# household_power: walk-forward test, 329 origins x 24 h

| model | mae | rmse | mase | seeds |
|---|---|---|---|---|
| seasonal_naive | 0.5541 | 0.8189 | 0.8100 | 1 |
| weekly_naive | 0.5549 | 0.8162 | 0.8113 | 1 |
| ets | 0.5794 | 0.7739 | 0.8471 | 1 |
| sarima | 0.4462 | 0.6221 | 0.6523 | 1 |
| lstm | 0.4382 ± 0.0033 | 0.6376 ± 0.0060 | 0.6406 ± 0.0048 | 3 |
| transformer | 0.4369 ± 0.0000 | 0.6295 ± 0.0055 | 0.6388 ± 0.0001 | 3 |

Deterministic models (naive, ETS, SARIMA) run once, so they show no ±.

## Cost

| model          |   seed | params   | best_epoch   |   train_s |   predict_ms_per_origin | fallbacks   |
|:---------------|-------:|:---------|:-------------|----------:|------------------------:|:------------|
| seasonal_naive |      0 |          |              |     nan   |                     0.0 |             |
| weekly_naive   |      0 |          |              |     nan   |                     0.0 |             |
| ets            |      0 |          |              |     nan   |                    73.2 | 0           |
| sarima         |      0 |          |              |     nan   |                  1987.1 | 0           |
| lstm           |      0 | 53016    | 22           |     807.7 |                     2.6 |             |
| lstm           |      1 | 53016    | 14           |     578.3 |                     2.6 |             |
| lstm           |      2 | 53016    | 12           |     518.9 |                     2.4 |             |
| transformer    |      0 | 114328   | 9            |    3220.4 |                     3.4 |             |
| transformer    |      1 | 114328   | 10           |    3405.0 |                     2.4 |             |
| transformer    |      2 | 114328   | 9            |    3165.8 |                     2.2 |             |

## Difference in MAE vs seasonal naive (paired bootstrap over forecast days, first seed)

| model        |   mae_diff_kw |   ci95_lo |   ci95_hi |   days_better |
|:-------------|--------------:|----------:|----------:|--------------:|
| weekly_naive |        0.0008 |   -0.0268 |    0.0268 |        0.4711 |
| ets          |        0.0253 |    0.0013 |    0.0530 |        0.4894 |
| sarima       |       -0.1079 |   -0.1256 |   -0.0918 |        0.7751 |
| lstm         |       -0.1149 |   -0.1342 |   -0.0976 |        0.7295 |
| transformer  |       -0.1172 |   -0.1369 |   -0.0996 |        0.7599 |

Negative = lower MAE than seasonal naive.

## MAE (kW) by horizon step

| model          |    h=1 |    h=6 |   h=12 |   h=18 |   h=24 |
|:---------------|-------:|-------:|-------:|-------:|-------:|
| seasonal_naive | 0.3924 | 0.2051 | 0.5983 | 0.6890 | 0.5294 |
| weekly_naive   | 0.4296 | 0.2182 | 0.6320 | 0.6633 | 0.5363 |
| ets            | 0.3587 | 0.4471 | 0.6038 | 0.6486 | 0.5318 |
| sarima         | 0.2539 | 0.1771 | 0.4668 | 0.5503 | 0.4563 |
| lstm           | 0.2829 | 0.1634 | 0.5034 | 0.5133 | 0.4622 |
| transformer    | 0.2520 | 0.1825 | 0.5215 | 0.4859 | 0.4285 |
