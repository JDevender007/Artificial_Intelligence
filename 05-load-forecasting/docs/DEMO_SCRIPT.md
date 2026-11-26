# P5 60-second demo script

1. **(0-10 s)** "This forecasts the next 24 hours of one household's electricity use, from four years of minute readings. Every model is tested the same way: 329 days of 2010, each forecast from midnight using only the past."
2. **(10-25 s)** Show the results table. "Copying yesterday gives 0.55 kW mean error. SARIMA, an LSTM and a small Transformer all cut that by about 20%, to 0.44. A day-level bootstrap says that gain is real. Between those three, the difference is within noise."
3. **(25-35 s)** Show the cost table. "So I'd ship the LSTM. It's as accurate as the Transformer, trains 5× faster on CPU, and predicts in 2.5 ms, where SARIMA takes 2 seconds per forecast."
4. **(35-50 s)** Show the hour-of-day table and the worst days. "Error tracks behaviour. Night hours are easy; the evening peak and weekends are hard. The worst days are the household doing something unusual: a heavy day, or an empty house."
5. **(50-60 s)** Show the anomaly table. "Reusing the forecast for anomaly detection: a plain z-score misses every injected power dropout, because zero is normal at night. Against the forecast, 85–95% of events are caught. Precision is still low, so it's a review list, not an alarm."
