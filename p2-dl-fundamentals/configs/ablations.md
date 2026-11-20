# Ablations (CNN on CIFAR-10, 3 seeds each)

Each ablation turns off one component of the main config and writes to its own folder, so
results never mix. Compare `cnn` rows against `outputs/cifar10/results.md`.

| Ablation | Command |
|---|---|
| No augmentation | `python scripts/run_experiment.py --config configs/cifar10.yaml --set name=no_aug output_dir=outputs/abl_no_aug "models=[cnn]" data.augment=false` |
| No LR schedule (constant LR) | `python scripts/run_experiment.py --config configs/cifar10.yaml --set name=no_sched output_dir=outputs/abl_no_sched "models=[cnn]" train.scheduler=none train.lr=0.01` |
| No mixed precision (speed + accuracy) | `python scripts/run_experiment.py --config configs/cifar10.yaml --set name=no_amp output_dir=outputs/abl_no_amp "models=[cnn]" train.amp=false` |
| ResNet-18 from scratch (no pretraining) | not wired by default: `build_model(..., pretrained=False)`; run as a stretch goal |

Then build the table: `python scripts/compare_ablations.py`.

What each one tests:
- **no_aug**: how much of the CNN's generalisation comes from crop/flip. Expect a larger train/val gap.
- **no_sched**: OneCycle vs constant LR at a safe value. Expect slower convergence and a lower final accuracy.
- **no_amp**: fp32 vs fp16 autocast. Expect similar accuracy and lower `images_per_s`. The throughput difference is the point.
