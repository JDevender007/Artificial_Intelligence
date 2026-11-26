# P11 interview Q&A

**1. Walk me through what happens between a commit and a deployed model.**
- **On commit,** CI runs ruff and the 11 pytest tests on synthetic data. It then trains a 1-seed model, builds the Docker image, starts it, and runs the checks:
  - curl smoke tests on `/ready`, `/model-info` and `/predict`, plus a 422 check on an invalid body;
  - a 30-second Locust run;
  - `docker stats`.
  The reports are uploaded as artifacts.
- **Locally or in a scheduled job,** `dvc repro` rebuilds only the stages whose dependencies or params changed (prepare → train → drift). MLflow logs every run, and the gate decides whether the GBDT replaces the baseline. The winner is registered as `credit-default@champion`.
- **Deploy** is a manual `workflow_dispatch`. It retrains from `params.yaml` and pushes the image context to a free Hugging Face Docker Space.

**2. Why a promotion gate, and why those two conditions?**
- **PR-AUC margin above 1 × seed std:** a candidate that is better only within seed noise isn't worth the extra complexity. LightGBM's margin is 0.057 against a seed std of 0.0025, so this is a clear pass.
- **ECE ≤ 0.03:** a reviewer reads the probability, so a model that ranks better but is miscalibrated is a regression for this product.
- **Single-seed edge case:** with one seed the std is NaN, so `NaN > x` is False. The gate silently never promoted until I made that case explicit (std = 0, with a warning).

**3. How do you stop training/serving skew?**
- **One pickled pipeline:** the feature transformer is row-wise and stateless, and it lives inside the pickled model, so the API runs the exact same code path.
- **One shared schema:** a single Pydantic schema defines the 19 inputs, with ranges and `extra="forbid"`. The service refuses to start if the model's feature list differs from it.
- **Pinned versions:** the image pins the same scikit-learn and LightGBM versions as training. That's necessary because the artefact is a pickle.

**4. Your drift monitor didn't fire in the recession drill. Isn't that a failure?**
It's the most useful result in the project.
- **The miss:** in the recession scenario, input distributions barely move (top PSI 0.10, 0/19 columns drifted), but the default rate goes from 0.23 to 0.40. ECE jumps from 0.013 to 0.142: the model under-predicts by 14 points.
- **The false alarm:** the limit-cut scenario moves one input a lot (PSI 0.52) and hardly hurts the model.
- **The lesson:** input drift is neither necessary nor sufficient for performance drift. The fixes are:
  - alert on **score** PSI, which did move to 0.127;
  - track ECE on matured, labelled cohorts, since card defaults are observable after a month;
  - keep a recalibration path ready.
- **What I didn't do:** retune the thresholds until the drill went red.

**5. How would this scale, and what does it cost?**
- **Workload:** the service is CPU-only and stateless, so throughput scales with workers and replicas.
- **Where a request's time goes:** a single prediction costs about 20 ms in-process (`reports/latency_profile.md`).
  - About 9 ms is pandas feature engineering, which is fixed per call.
  - About 10 ms is the classifier: five calibrated LightGBM folds averaged.
  - 1,000 rows cost only about 72 ms, so batching amortises almost everything.
- **Cheap wins:**
  - a NumPy feature path;
  - one booster instead of five calibration folds (refit once, then calibrate on a holdout);
  - `n_jobs=1` per booster, so threads don't fight with the uvicorn workers.
- **Measurement:** I measured throughput with Locust (README §F) and costed it from published price pages:
  - Hugging Face CPU Basic is free;
  - Cloud Run bills per vCPU-second and GiB-second, with a monthly free tier.
- **Next steps for production:**
  - put the prediction log on a queue or object storage rather than a local file;
  - run the monitor as a scheduled job;
  - add a shadow deployment for a new champion before switching the alias.
