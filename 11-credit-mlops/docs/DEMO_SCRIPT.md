# P11 60-second demo script

1. **(0-10 s)** Show the README diagram. "This is my credit-default model taken to production shape: a DVC pipeline, an MLflow registry with a promotion gate, a FastAPI service in Docker, CI that load-tests the image, and drift monitoring."
2. **(10-22 s)** Run `dvc repro`, then show `reports/results.md`. "Logistic baseline vs calibrated LightGBM, three seeds. The gate promotes LightGBM only because its PR-AUC margin is 23× its seed std *and* its calibration error is under 0.03."
3. **(22-35 s)** Run `mlflow ui` and show the `credit-default` model with the `champion` alias. Then open `localhost:8000/docs` and send one `/predict`. "Probability, a flag at the out-of-fold threshold, and the model version. An invalid field gets a 422. `/metrics` exposes the latency histogram."
4. **(35-47 s)** Show the Locust summary in `reports/locust/`. "Here are the measured requests per second and p95 latency on a laptop, and the cost estimate built from them in `reports/cost.md`."
5. **(47-60 s)** Show `reports/drift.md`. "The finding: a simulated recession moves no input past the drift rule, yet calibration error goes from 0.013 to 0.142. Input drift monitoring alone would have stayed green. The score PSI moved, and label-delayed calibration checks are the real safety net."
