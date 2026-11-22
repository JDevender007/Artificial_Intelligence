# Head-to-head (paired bootstrap over forecast days; negative = b has lower MAE)

| a      | b           |   seed |   mae_diff_b_minus_a |   ci95_lo |   ci95_hi |   days_b_better |
|:-------|:------------|-------:|---------------------:|----------:|----------:|----------------:|
| sarima | transformer |      0 |              -0.0093 |   -0.0219 |    0.0033 |          0.5167 |
| sarima | transformer |      1 |              -0.0093 |   -0.0206 |    0.0032 |          0.5319 |
| sarima | transformer |      2 |              -0.0092 |   -0.0210 |    0.0034 |          0.5350 |
| sarima | lstm        |      0 |              -0.0070 |   -0.0202 |    0.0070 |          0.5562 |
| sarima | lstm        |      1 |              -0.0053 |   -0.0189 |    0.0093 |          0.5228 |
| sarima | lstm        |      2 |              -0.0117 |   -0.0251 |    0.0019 |          0.5258 |
| lstm   | transformer |      0 |              -0.0023 |   -0.0109 |    0.0067 |          0.4650 |
| lstm   | transformer |      1 |              -0.0040 |   -0.0106 |    0.0024 |          0.5319 |
| lstm   | transformer |      2 |               0.0024 |   -0.0046 |    0.0093 |          0.4802 |

## Seed-averaged ensembles vs SARIMA

| model                     |    mae |   rmse |   vs_sarima |   ci95_lo |   ci95_hi |   days_better |
|:--------------------------|-------:|-------:|------------:|----------:|----------:|--------------:|
| lstm (3-seed mean)        | 0.4295 | 0.6283 |     -0.0167 |   -0.0298 |   -0.0030 |        0.5714 |
| transformer (3-seed mean) | 0.4278 | 0.6188 |     -0.0185 |   -0.0304 |   -0.0064 |        0.5745 |
