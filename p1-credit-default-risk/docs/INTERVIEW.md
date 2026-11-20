# Interview questions: P1 credit default

**1. Why PR-AUC as the primary metric instead of accuracy or ROC-AUC?**
About 22% of clients default. Predicting "no default" for everyone already gives ~78% accuracy, so accuracy says little. ROC-AUC counts true negatives, and with many negatives it can look good while precision on the default class is poor. PR-AUC only uses precision and recall on the positive class, so it tracks what a lender cares about: catching defaulters without too many false alarms. A random model's PR-AUC equals the prevalence, which gives a clear floor to compare against.

**2. How did you prevent data leakage?**
(a) The test split is stratified, made once with a fixed seed and used once per model. (b) All preprocessing sits inside one sklearn `Pipeline`, so cross-validation refits the scalers and encoders on each training fold. (c) Feature engineering is row-wise and stateless; a unit test checks that what it is fitted on does not change its output. (d) Optuna tuning and F1-threshold selection use out-of-fold train predictions only. (e) An automated check flags any single feature with ROC-AUC ≥ 0.9 and counts duplicate rows shared between train and test.

**3. Why calibrate, and how?**
`scale_pos_weight` and class weights push predicted probabilities upward, so a score of 0.6 is not a 60% default risk. Lending decisions such as pricing, expected loss (PD × LGD × EAD) and limits need real probabilities. I wrapped the tuned model in `CalibratedClassifierCV(method="isotonic", cv=5)` and measured ECE and Brier score before and after. Isotonic is monotone, so ranking metrics barely change. Platt (sigmoid) scaling is better when data is small, because isotonic can overfit.

**4. Why drop SEX, AGE, MARRIAGE and EDUCATION, and is that enough for fairness?**
In lending, using these as inputs risks disparate treatment, and regulators restrict it. Dropping them is "fairness through unawareness". That is **not** enough, because proxies such as credit limit or spending patterns can still correlate with the dropped attributes. That is why the error analysis measures TPR, FPR and selection-rate gaps for each group, and an ablation measures what the attributes would add.

**5. What does SHAP tell you here, and what are its limits?**
TreeSHAP splits each prediction (in log-odds) into additive feature contributions. The global mean |SHAP| ranks features, and the beeswarm plot shows the direction of each effect. Limits: it explains the *model*, not causation. Correlated features share credit in ways that can mislead. It also explains the uncalibrated booster, which is fine for ranking because isotonic calibration is monotone.

*(Prepare one extra answer from your own results: "What would you do next?" Use the error analysis, e.g. "silent defaulters" with clean recent history, plus a time-based split if dated data were available.)*
