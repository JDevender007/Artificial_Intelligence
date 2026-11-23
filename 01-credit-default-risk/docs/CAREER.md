# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/01-credit-default-risk` (monorepo). Standalone name if split out: `credit-default-risk-ml`

## Resume bullet (numbers from `outputs/default/results.md`)

- Built a leakage-safe credit-default model on 30K UCI clients (scikit-learn pipelines, Optuna-tuned LightGBM/XGBoost). Reached test PR-AUC **0.557 Â± 0.001** vs **0.496 Â± 0.000** for a tuned logistic-regression baseline (3 seeds), and cut calibration error (ECE) from **0.200** to **0.015** with isotonic calibration.
- Added SHAP explanations, a slice-level error analysis and a fairness audit (TPR/FPR gaps by sex and age, with demographic inputs excluded), backed by pytest and GitHub Actions CI. The full experiment reruns from one command.

## LinkedIn post draft

> Project 1/11 of my AI-engineering portfolio: **credit default prediction**.
>
> Most tutorials stop at "XGBoost got 82% accuracy". On this dataset, predicting "nobody defaults" already gets ~78%, so I tried to do it properly:
> - PR-AUC as the main metric, with mean Â± std over 3 seeds on one fixed test set
> - Leakage checks before any training, and all preprocessing inside CV folds
> - A tuned logistic-regression baseline (it's harder to beat than you'd think): **0.496** vs my model's **0.557** PR-AUC
> - Isotonic calibration: ECE **0.200 â†’ 0.015**, because a lender needs real probabilities, not just a ranking
> - SHAP and a fairness audit: demographics excluded from inputs, error gaps measured per group
>
> Biggest lesson: **the model is mostly a 'were you late last month?' detector. Defaulters who paid on time until the day they stopped are almost invisible in this data**.
>
> Code + write-up: https://github.com/JDevender007/Artificial_Intelligence/tree/main/01-credit-default-risk  #MachineLearning #DataScience #XGBoost #ResponsibleAI
