# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/11-credit-risk-mlops` (monorepo). Standalone name if split out: `credit-risk-mlops`

**Resume bullet**
- Productionised a calibrated LightGBM credit-default model:
  - a DVC pipeline;
  - MLflow tracking and a registry with a seed-noise and calibration promotion gate (PR-AUC 0.555 vs 0.498 baseline, ECE 0.009);
  - a typed FastAPI service with Prometheus metrics, in Docker;
  - GitHub Actions that train, build, smoke-test and Locust-load-test the image;
  - a free Hugging Face deploy.
- Drift drills showed input-PSI monitoring misses a recession-style concept shift (ECE 0.013 → 0.142 with 0/19 columns drifted). The monitor now reports score PSI, and the write-up specifies the fix: alert on score PSI and run label-delayed calibration checks.

**LinkedIn post (draft)**
> My drift monitor stayed green while the model's probabilities went wrong by 14 points.
>
> For the last project in my portfolio I took my credit-default model to production shape: DVC pipeline, MLflow registry with a promotion gate, FastAPI + Docker, GitHub Actions that build and load-test the image, Evidently reports, and a free Hugging Face deploy.
>
> Then I ran drift drills on held-out data:
> - Cut every credit limit by 30%: big input drift (PSI 0.52), and the model barely notices.
> - Simulate a recession (more late payments, higher bills, default rate 23% → 40%): almost no input drift, 0 of 19 columns flagged. But calibration error jumps 0.013 → 0.142.
>
> Input drift is neither necessary nor sufficient for model failure. What did move was the *score* distribution (PSI 0.13). The real safety net is checking calibration on matured, labelled cohorts.
>
> I didn't tune the thresholds to make the demo go red. The miss is the lesson.
>
> Code, drills and CI: [repo link]
