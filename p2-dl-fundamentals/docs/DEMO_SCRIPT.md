# 60-second demo script (P2)

**0–10 s: the hook.** "This is a CNN and an MLP I wrote from scratch in PyTorch: my own linear layer, my own training loop. They're compared against a pretrained ResNet-18 on CIFAR-10, over three seeds."

**10–25 s: the code.** Open `src/dlfund/models.py` and show `ScratchLinear` (one line: `x @ W.T + b`, He init). Then `train.py`: the loop with autocast, GradScaler and a OneCycle step per batch. "No Trainer, no Lightning: every step is visible."

**25–40 s: the run.** Start `python scripts/run_experiment.py --config configs/smoke.yaml` and show TensorBoard (`tensorboard --logdir outputs/smoke/tb`): train vs val loss and the LR curve. Kill it with Ctrl-C and rerun: "It resumes from `last.pt`, including the scheduler and RNG state."

**40–55 s: the results.** Open `reports/cifar10/results.md`: accuracy mean ± std for the linear baseline, MLP, CNN and ResNet-18, plus params and images/s. Then the CNN confusion matrix: "Most mistakes are cat vs dog, and truck vs automobile."

**55–60 s: the close.** "34 CPU tests run in CI on fake data, and every number reproduces from one command."

_Before recording, check that the confusion claim matches `per_class.md`. Edit the line if your run shows different top confusions._
