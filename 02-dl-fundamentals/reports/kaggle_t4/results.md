# Results: CIFAR-10 and MNIST on a Kaggle T4

Test set, mean ± std over seeds 0, 1, 2. Every row is parsed from the raw kernel logs in this
folder (`cifar10.log`, `mnist.log`, `abl_*.log`); nothing here is typed by hand.

Reproduce with `python scripts/run_experiment.py --config configs/<config>.yaml` on a GPU.

| dataset | model | test acc | macro-F1 | test loss | params (M) | train min | img/s |
|---|---|---|---|---|---|---|---|
| MNIST | linear (baseline) | 0.8502 ± 0.0041 | 0.8477 ± 0.0046 | 0.5601 ± 0.0049 | 0.01 | 1.8 | 7,041 |
| MNIST | mlp | 0.9883 ± 0.0003 | 0.9882 ± 0.0003 | 0.0367 ± 0.0011 | 0.5 | 2.0 | 6,791 |
| MNIST | cnn | **0.9964 ± 0.0006** | 0.9964 ± 0.0006 | 0.0105 ± 0.0010 | 1.1 | 2.2 | 6,234 |
| CIFAR-10 | linear (baseline) | 0.3266 ± 0.0042 | 0.3101 ± 0.0062 | 1.9625 ± 0.0095 | 0.03 | 0.8 | 5,599 |
| CIFAR-10 | mlp | 0.5752 ± 0.0034 | 0.5697 ± 0.0030 | 1.1887 ± 0.0035 | 1.7 | 4.1 | 5,458 |
| CIFAR-10 | cnn | 0.9055 ± 0.0005 | 0.9054 ± 0.0005 | 0.3086 ± 0.0031 | 1.1 | 4.3 | 5,179 |
| CIFAR-10 | resnet18 (ImageNet-pretrained, 128 px) | **0.9535 ± 0.0011** | 0.9534 ± 0.0011 | 0.1450 ± 0.0014 | 11.2 | 2.9 | 2,076 |

Ablations (from-scratch CNN, CIFAR-10, 3 seeds; one change each vs the full recipe):

| variant | test acc | Δ vs full | train min | img/s |
|---|---|---|---|---|
| full recipe (aug + OneCycle + AMP) | 0.9055 ± 0.0005 | – | 4.3 | 5,179 |
| no mixed precision (fp32) | 0.9083 ± 0.0027 | +0.3 (tie) | 9.0 | 2,499 |
| no augmentation | 0.8158 ± 0.0164 | −9.0 | 1.8 | 7,112 |
| no OneCycle (constant LR) | 0.8323 ± 0.0225 | −7.3 | 3.3 | 5,231 |
