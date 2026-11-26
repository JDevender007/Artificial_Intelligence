# Interview Q&A (P2)

**1. Why write the linear layer and training loop yourself instead of using `nn.Linear` and a Trainer?**
To show I know what they hide. `ScratchLinear` is `x @ W.T + b` with He-normal init (std = sqrt(2 / fan_in), because ReLU zeroes half the activations). A unit test checks it equals `F.linear` exactly. The loop makes every step visible: autocast forward, scaled backward, unscale, optional gradient clipping, `scaler.step`, then a per-batch scheduler step. In real work I'd use `nn.Linear`; the point is being able to debug it when a framework misbehaves.

**2. How does mixed precision work, and why do you need GradScaler?**
`torch.autocast` runs matmuls and convolutions in fp16 and keeps reductions such as softmax and the loss in fp32. fp16 has a small exponent range, so small gradients underflow to zero. GradScaler multiplies the loss by a large factor before backward, unscales before the optimiser step, skips steps with inf/NaN gradients, and adapts the factor. On bf16 hardware you don't need it. The `no_amp` ablation measures the speed and accuracy difference on my GPU.

**3. How do you make a deep-learning run reproducible, and what are the limits?**
Seed Python, NumPy and torch. Give the DataLoader a seeded generator and seed each worker. Use a fixed split seed, separate from the training seed, so every seed sees the same validation set. Then cuDNN autotuning (`benchmark=True`) and atomic adds on the GPU are still non-deterministic. `deterministic=true` turns on `use_deterministic_algorithms` for exact repeats at a speed cost. I report mean ± std over seeds because seed variance is real signal, not noise to hide.

**4. Why early-stop on validation loss and not accuracy? Why test only once?**
Loss is continuous: it detects growing over-confidence before accuracy moves, and it's less noisy on a 5k validation set. The test set is touched once, with the best-val-loss weights. If I picked epochs or hyperparameters by test accuracy, the reported number would be optimistically biased; that is the same leakage argument as in P1.

**5. The pretrained ResNet beats your CNN. What does that prove?**
On its own, not much: it has ~10× the parameters, sees 128 px upsampled inputs and has ImageNet features. Two effects are mixed: pretraining and capacity. The clean test is a ResNet-18 trained from scratch at the same resolution (an ablation in `configs/ablations.md`). In practice, pretrained backbones win when labelled data is small. For 32 px images and a tight latency budget, a small CNN can be the better engineering choice. The `images_per_s` column shows that trade-off.
