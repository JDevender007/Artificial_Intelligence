# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/05-load-forecasting` (monorepo). Standalone name if split out: `load-forecasting`

**Resume bullet**
- Built a day-ahead household load forecaster (UCI, 34.6k hours) with a 329-origin walk-forward backtest comparing naive, ETS, SARIMA, LSTM and Transformer models over 3 seeds with a day-level paired bootstrap. The LSTM/Transformer/SARIMA tier cut MAE about 20% vs seasonal naive (0.437 vs 0.554 kW). Forecast-residual anomaly detection raised injected-event recall from 0.53 to 0.85–0.95.

**LinkedIn post (draft)**
> I trained a Transformer to forecast a household's electricity use. It won by 0.001 kW, and I'm not shipping it.
>
> Setup: 4 years of one house's readings, a 24-hour-ahead forecast every midnight across all of 2010, models only ever see the past.
> - Copy yesterday: 0.554 kW mean error.
> - SARIMA 0.446, LSTM 0.438, Transformer 0.437. All about 20% better than the baseline, and within noise of each other.
> - The LSTM trains 5× faster on CPU than the Transformer, so that's the one to deploy.
>
> Bonus: using forecast residuals for anomaly detection. A plain z-score missed every simulated power outage, because 0 kW is normal at night. Against the forecast, 85–95% of events were caught.
>
> Code, backtest and error analysis: [repo link]
