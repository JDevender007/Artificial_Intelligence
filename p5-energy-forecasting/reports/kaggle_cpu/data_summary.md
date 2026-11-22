| split | hours | start | end | mean kW | std kW | p95 kW |
|---|---|---|---|---|---|---|
| train | 22,248 | 2006-12-17 00:00:00 | 2009-06-30 23:00:00 | 1.118 | 0.944 | 2.994 |
| val | 4,416 | 2009-07-01 00:00:00 | 2009-12-31 23:00:00 | 1.007 | 0.831 | 2.657 |
| test | 7,896 | 2010-01-01 00:00:00 | 2010-11-25 23:00:00 | 1.052 | 0.782 | 2.571 |

Train peak hour 20:00, trough 3:00. Autocorrelation at lag 1 / 24 / 168 h: 0.718 / 0.425 / 0.453.

Cleaning: 25,979 of 2,075,259 minute readings missing; 421 hours had no reading, 1 were interpolated (gaps of 6 h or less) and 420 were filled from one week earlier. Series: 2006-12-17 00:00:00 to 2010-11-25 23:00:00 (34,560 hours).
