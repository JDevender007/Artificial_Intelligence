# P5 interview questions

**1. The Transformer had the lowest MAE. Why recommend the LSTM?**
The three good models sit within 0.01 kW: Transformer 0.4369, LSTM 0.4382 ± 0.0033, SARIMA 0.4462. A paired day-level bootstrap puts every pairwise CI across 0 (Transformer vs SARIMA −0.009 kW, CI [−0.022, +0.003]), so for a single model the ranking is noise. At that point cost decides. On the same CPU the LSTM trained in 9–13 minutes per seed against about 54 for the Transformer, with half the parameters (53k vs 114k). Both predict in about 2.5 ms, while SARIMA needs 2 s per forecast because it refits. The metric also matters: SARIMA has the best RMSE (0.622), so if peak misses cost more than average error, SARIMA is a defensible choice.

**2. How did you make sure the backtest doesn't leak?**
- Every origin is a midnight in 2010, and the model receives `y.iloc[:t0]`, history strictly before the origin. `test_no_lookahead` checks this, and `test_future_corruption_does_not_change_forecasts` sets everything after the first forecast window to 1e6 and asserts that the first forecast does not change.
- The statistical models refit only on the 28 days before each origin.
- The neural models train only on data before 2009-07 and use the second half of 2009 for early stopping, so 2010 is untouched until scoring.
- Gap filling uses the value from one week *earlier*, never later.
- The anomaly z-score uses only the previous window's median and MAD.

**3. Why does the MAE-by-horizon curve go down at step 6 and up at step 18? Shouldn't error grow with horizon?**
Because every origin is midnight, step *h* is always clock hour *h*−1. The curve is the daily usage cycle: 05:00 is easy (everyone asleep), 17:00–21:00 is hard (cooking, appliances). With this design, horizon decay and time-of-day difficulty are confounded. To separate them I would use origins at every hour, and compare the same clock hour at different horizons.

**4. Why does a raw z-score miss every dropout, and what fixed it?**
A dropout sets the load to about 0 kW for 3–12 hours, but near-zero is a normal night value, so the raw value is not unusual. The residual (actual minus forecast) is: 0 kW when the model expected 1.5 kW is a large deviation. Event recall went from 0.53 to 0.85–0.95 with residual detectors. The best of them, LSTM residuals, has F1 0.26 and about 10 false alarms a week. Precision is still about 0.19, so I would add a duration rule and use it as a review queue, not a pager.

**5. ETS is worse than seasonal naive on MAE but better on RMSE. How?**
ETS forecasts a smoothed day. At night it over-forecasts (03:00–05:00 MAE about 0.43 vs 0.21 for naive), which costs MAE. At the evening peak, a naive copy of yesterday is sometimes badly wrong when the peak shifts, and the squared error of those misses dominates RMSE; ETS's blurred forecast avoids the worst of them. The two metrics answer different questions: typical error vs worst misses. That is why I report both and state which one the recommendation is based on.
