# AI / ML Engineering Portfolio

Eleven end-to-end projects, from classical ML to MLOps. Every project follows the same rules:

- **Honest numbers:** every result is from a real run, reported as mean ± std over 3+ seeds against a simple baseline. Nothing is copied from papers.
- **Reproducible:** fixed seeds and one command per experiment. Pinned requirements, with a lock file.
- **Engineered:** a `src/` package (notebooks only call it), type hints, logging, pytest, and lint + test CI per project.
- **Responsible:** dataset licences, privacy and fairness notes in each README.

| # | Project | Topic | Headline result | CI |
|---|---|---|---|---|
| 1 | [Credit default risk](p1-credit-default-risk/) | Tabular ML: leakage checks, Optuna, calibration, SHAP, fairness audit | Test PR-AUC 0.557 ± 0.001 vs 0.496 for a tuned logistic baseline; ECE 0.200 → 0.015 after isotonic calibration | ![p1](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p1-ci.yml/badge.svg) |
| 2 | [DL fundamentals](p2-dl-fundamentals/) | MLP/CNN from scratch in PyTorch, own training loop, vs a ResNet-18 fine-tune | code + tests done; results pending (Kaggle run) | |
| 3 | [PPE detection + tracking](p3-ppe-detection-tracking/) | YOLO26 scratch vs pretrained + ablations, own error analysis, per-person compliance rule, ByteTrack vs BoT-SORT with own CLEAR-MOT | code + tests done; results pending (Kaggle run) | ![p3](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p3-ci.yml/badge.svg) |
| 4 | [Banking77 intent classification](p4-intent-classification/) | TF-IDF + LR vs DistilBERT, macro-F1 / ECE / latency, error analysis by class and length, model card | TF-IDF + LR: test macro-F1 0.913 at 2.3 ms/message on CPU; DistilBERT pending (Kaggle run) | ![p4](https://github.com/JDevender007/Artificial_Intelligence/actions/workflows/p4-ci.yml/badge.svg) |
| 5 | Time series | forecasting with walk-forward backtest + anomaly detection | planned | |
| 6 | Recommender systems | MF vs two-tower, FAISS serving | planned | |
| 7 | RAG + QLoRA | retrieval evaluation, 4-bit fine-tune | planned | |
| 8 | Agents | tool-using agent with a 30-task eval | planned | |
| 9 | Multimodal | speech + vision assistant | planned | |
| 10 | Reinforcement learning | DQN/PPO, 5 seeds with confidence bands | planned | |
| 11 | MLOps | FastAPI, Docker, MLflow, DVC, CI/CD, monitoring | planned | |

## Layout

Each `pN-*/` folder is a self-contained project with its own `.venv`, `requirements.txt`, tests and README. Data and run outputs are gitignored; each project's README gives the command that regenerates them, and a small `reports/` snapshot of the final run is committed.

Compute used: one laptop (RTX 3050 4 GB) plus free Kaggle/Colab GPUs.
