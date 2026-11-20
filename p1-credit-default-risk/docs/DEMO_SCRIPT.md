# 60-second demo video script

| Time | Screen | Voice-over |
|---|---|---|
| 0–8s | README title + Mermaid diagram | "This project predicts whether a credit-card client defaults next month, on 30,000 real anonymised clients from Taiwan." |
| 8–18s | Terminal: `python scripts/run_experiment.py --config configs/smoke.yaml` scrolling, leakage report line | "One command runs everything: leakage checks, a tuned logistic-regression baseline, Optuna-tuned XGBoost and LightGBM, and calibration, over three seeds." |
| 18–30s | `results.md` table, highlight baseline vs final row | "On the same held-out test set the final model reaches PR-AUC of [X ± s] vs [Y ± s] for the baseline, and calibration cuts the expected calibration error from [A] to [B]." |
| 30–40s | `reliability.png`, then `shap_beeswarm.png` | "Calibrated probabilities matter for lending decisions. SHAP shows repayment status last month and the repayment ratio drive risk." *(say only what YOUR plot shows)* |
| 40–52s | `error_analysis.md` slice table and fairness gaps | "The model misses 'silent defaulters' who paid on time last month. Demographics are excluded from inputs, and I audit TPR and FPR gaps across sex and age." *(use your numbers)* |
| 52–60s | GitHub repo with green CI badge | "Everything is tested, reproducible from one command, and open source. Link below." |

Recording tips: OBS Studio (free), 1080p, terminal font ≥ 16pt, run the smoke config live and cut to the pre-computed full results.
