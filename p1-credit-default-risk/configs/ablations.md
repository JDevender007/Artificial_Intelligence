# Ablations (run with `--set`, each into its own output_dir)

All other settings stay at `configs/default.yaml`. Compare `results.md` across folders.

| Question | Command |
|---|---|
| Do engineered features help? | `python scripts/run_experiment.py --config configs/default.yaml --set output_dir=outputs/abl_no_fe features.engineered=false` |
| Does class re-weighting help (PR-AUC, calibration)? | `python scripts/run_experiment.py --config configs/default.yaml --set output_dir=outputs/abl_no_weight imbalance=none` |
| What do demographics add (and at what fairness cost)? | `python scripts/run_experiment.py --config configs/default.yaml --set output_dir=outputs/abl_demographics features.drop_protected=false` |
| Isotonic vs Platt calibration | `python scripts/run_experiment.py --config configs/default.yaml --set output_dir=outputs/abl_sigmoid calibration=sigmoid` |

Tip: ablations can use `tuning.n_trials=25` to save time, but then compare against a
default run with the same trial budget.
