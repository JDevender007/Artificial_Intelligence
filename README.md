# AI / ML Engineering Portfolio

Eleven end-to-end projects, from classical ML to MLOps. Every project follows the same rules:

- **Honest numbers:** every result is from a real run, reported as mean ± std over 3+ seeds against a simple baseline. Nothing is copied from papers or invented.
- **Reproducible:** fixed seeds and one command per experiment. Pinned requirements in every project, plus a full `requirements-lock.txt` freeze where the environment was captured (P01).
- **Engineered:** a `src/` package (notebooks only call it), type hints, logging, pytest, and lint + test CI per project.
- **Responsible:** dataset licences, privacy and fairness notes in each README.

| # | Project | Topic | Headline result | CI |
|---|---|---|---|---|
| 1 | [Credit-risk scoring](01-credit-risk-scoring/) | Tabular ML: leakage checks, Optuna, calibration, SHAP, fairness audit | Test PR-AUC 0.557 ± 0.001 vs 0.496 for a tuned logistic baseline; ECE 0.200 → 0.015 after isotonic calibration | ![p1](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p1-ci.yml/badge.svg) |
| 2 | [Vision from scratch](02-vision-from-scratch/) | MLP/CNN from scratch in PyTorch, own training loop, vs a ResNet-18 fine-tune | CIFAR-10 test acc: own CNN 0.9055 ± 0.0005 (1.1M params), pretrained ResNet-18 0.9535 ± 0.0011; AMP 2.1× throughput at equal accuracy | ![p2](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p2-ci.yml/badge.svg) |
| 3 | [PPE safety vision](03-ppe-safety-vision/) | YOLO26 scratch vs pretrained + ablations, per-person compliance rule, ByteTrack vs BoT-SORT with own CLEAR-MOT | pretrained YOLO26n: test mAP50-95 0.529 ± 0.003 vs 0.227 from scratch (3 seeds), 4.0 ms/img on T4; compliance agreement 0.696 | ![p3](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p3-ci.yml/badge.svg) |
| 4 | [Banking77 intent routing](04-intent-routing/) | TF-IDF + LR vs DistilBERT, macro-F1 / ECE / latency, error analysis by class and length, model card | DistilBERT 0.921 ± 0.006 vs TF-IDF 0.913 macro-F1 (bootstrap CI includes 0), ECE 0.023 vs 0.063; 29 vs 9 ms/request on the same CPU | ![p4](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p4-ci.yml/badge.svg) |
| 5 | [Household load forecasting](05-load-forecasting/) | naive / ETS / SARIMA / LSTM / Transformer under a 329-day walk-forward backtest, day-level bootstrap, residual anomaly detection | 24-h MAE: Transformer 0.437 ± 0.000, LSTM 0.438 ± 0.003, SARIMA 0.446 vs seasonal naive 0.554 kW; forecast residuals catch 85–95% of injected events vs 53% for a raw z-score | ![p5](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p5-ci.yml/badge.svg) |
| 6 | [MovieLens two-tower retrieval](06-two-tower-recsys/) | popularity vs BPR-MF vs feature two-tower (logQ, ablations), temporal split, FAISS HNSW, FastAPI cold-start | BPR-MF recall@20 0.097 beats two-tower 0.088 (bootstrap CI excludes 0); two-tower covers 6× more catalogue; HNSW 0.99 recall vs exact | ![p6](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p6-ci.yml/badge.svg) |
| 7 | [Open-book QA](07-openbook-qa/) | SQuAD 2.0 open-book QA: chunking ablation, BM25/dense/hybrid + reranker, citations, abstention, QLoRA 1.5B | **Results pending** the first completed GPU run — code, 15 tests and CI are done, no numbers claimed yet | ![p7](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p7-ci.yml/badge.svg) |
| 8 | [Guarded tool agent](08-guarded-tool-agent/) | Qwen2.5-7B NF4 agent: 8 tools, SQL/file/output guardrails, retries, tracing, 30-task eval | 0.878 ± 0.051 task success vs 0.244 ± 0.077 for direct prompting; guardrail violations 2.3 → 0 per episode | ![p8](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p8-ci.yml/badge.svg) |
| 9 | [Domain diffusion LoRA](09-domain-lora/) | Stable Diffusion 1.5 LoRA on 1,000 CC0 museum specimens: KID/CLIP, memorisation check, layered safety filter | **Results pending** the first completed GPU run — code, 17 tests and CI are done, no numbers claimed yet | ![p9](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p9-ci.yml/badge.svg) |
| 10 | [RL inventory control](10-rl-inventory-control/) | DQN from scratch + SB3 DQN/PPO on a custom Gymnasium inventory environment, 5 seeds with confidence bands | PPO 2,326 ± 5 ties the tuned (s,S) heuristic 2,331; from-scratch DQN 2,209 ± 103 | ![p10](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p10-ci.yml/badge.svg) |
| 11 | [Credit-risk MLOps](11-credit-risk-mlops/) | FastAPI + Docker + MLflow + DVC + GitHub Actions + Locust + Evidently, built on the P01 model | LightGBM PR-AUC 0.555 ± 0.003 vs 0.498 logistic baseline, ECE 0.009; promotion gate passed | ![p11](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p11-ci.yml/badge.svg) |

## Repository structure

```
Artificial_Intelligence/
├── .github/workflows/         one CI workflow per project (p1-ci.yml ... p11-ci.yml)
├── LICENSE                    MIT
├── README.md                  this file
├── .gitattributes             LF everywhere, binaries declared
├── .gitignore                 one ignore list for the whole monorepo
├── 01-credit-risk-scoring/
├── 02-vision-from-scratch/
├── ...
├── 11-credit-risk-mlops/
└── tools/kaggle/              kernel map + push script for Kaggle runs
```

Every project is a self-contained package with the same shape:

```
NN-capability-name/
├── README.md                  problem, data, approach, results, limitations
├── pyproject.toml             package metadata + ruff/pytest config
├── requirements.txt           runtime dependencies (pinned)
├── requirements-dev.txt       pytest + ruff
├── requirements-ci.txt        exactly what CI installs
├── requirements-gpu.txt       GPU training extras (only where a GPU is needed)
├── requirements-demo.txt      demo-app extras (only where a demo app needs them)
├── requirements-lock.txt      full freeze (only where the environment was captured)
├── requirements-serve.txt     serving image (P11 only)
├── configs/                   one YAML per experiment (smoke / full)
├── src/<package>/             all logic; notebooks only call this
├── scripts/                   one entry point per task (run_experiment, download, ...)
├── tests/                     pytest, CPU-only, no network
├── notebooks/                 thin Kaggle/Colab wrapper (not in 11: deployment project)
├── docs/
│   ├── CAREER.md              resume bullet, LinkedIn draft, repo name
│   ├── DEMO_SCRIPT.md         timed 60-second demo script
│   └── INTERVIEW.md           5 likely interview questions with answers
└── reports/                   committed snapshot of the final run
    └── <run>/results.md       one folder per run (multi-run projects)
```

Conventions:

- **The folder is the project name.** `NN-<capability>-<domain>` in portfolio order; the zero-padded number keeps the reading order stable (`10-` never sorts before `02-`). The standalone repo name drops the number and is recorded in each `docs/CAREER.md`.
- **`reports/` holds what was measured, never what was hoped.** Projects with a single committed snapshot keep the files at `reports/` root (P01, P11); projects with several runs use `reports/<run>/`.
- **`requirements-ci.txt` is the CI contract.** The lint+test job installs that file and nothing else, so a green badge means those pins work. Where CI is a strict subset of the runtime (no torch), the subset is written out explicitly instead of pulled in with `-r`.
- **`notebooks/` is a wrapper.** It clones the repo, installs `requirements-gpu.txt` and calls `scripts/run_experiment.py`; no logic lives in the notebook.
- **Data and run outputs are gitignored.** Each README gives the command that regenerates them.

## Compute

One laptop (RTX 3050 4 GB) for development and the app/deployment work, plus free Kaggle (T4 x2 / P100, ~9 h sessions) and Colab (free T4) for training. `tools/kaggle/` pushes each project's notebook and runs it there.

## Licence

MIT — see [LICENSE](LICENSE). Datasets keep their own licences, stated in each project README.
