# Anomaly detection on injected events (test period)

40 events per period (spike, dropout, level_shift); the z threshold is picked on a separate validation injection and applied unchanged here.

| detector       |     z |   precision |   recall |    f1 |   event_recall |   recall_spike |   recall_dropout |   recall_level_shift |   false_alarms_per_week |
|:---------------|------:|------------:|---------:|------:|---------------:|---------------:|-----------------:|---------------------:|------------------------:|
| raw_zscore     | 3.000 |       0.182 |    0.247 | 0.210 |          0.525 |          0.929 |            0.000 |                0.615 |                   6.979 |
| seasonal_naive | 2.500 |       0.143 |    0.407 | 0.212 |          0.950 |          1.000 |            0.923 |                0.923 |                  15.298 |
| weekly_naive   | 2.500 |       0.142 |    0.464 | 0.217 |          0.900 |          1.000 |            0.692 |                1.000 |                  17.617 |
| lstm           | 3.000 |       0.194 |    0.390 | 0.259 |          0.850 |          1.000 |            0.615 |                0.923 |                  10.149 |
