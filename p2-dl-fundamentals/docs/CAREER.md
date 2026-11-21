# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/p2-dl-fundamentals` (monorepo). Standalone name if split out: `pytorch-from-scratch-cnn`

## Resume bullet (fill in from `reports/*/results.md` once the runs finish)

- Implemented an MLP and CNN from scratch in PyTorch (custom linear layer, hand-written training loop with fp16 mixed precision, OneCycle LR, early stopping, resumable checkpoints, TensorBoard). Reached **90.6 ± 0.05%** CIFAR-10 test accuracy with 1.15M parameters, vs 57.5% for an MLP and 95.4% for a fine-tuned ResNet-18 (3 seeds, Kaggle T4).
- Ran ablations showing augmentation adds **9.0** points, the OneCycle schedule adds **7.3**, and mixed precision gives **2.1×** throughput at equal accuracy. 34 pytest tests run on CPU in GitHub Actions.

## LinkedIn post draft

> Project 2/11 of my AI-engineering portfolio: **deep learning without the training wheels**.
>
> I wrote the linear layer (`x @ W.T + b`) and the whole training loop myself in PyTorch: mixed precision, OneCycle LR, early stopping, checkpoint resume, TensorBoard. Then I compared:
> - softmax regression floor: **32.7%** (under-tuned; see README)
> - my MLP: **57.5%**
> - my 1.15M-param CNN: **90.6%**
> - pretrained ResNet-18 fine-tune: **95.4%**
>
> All on CIFAR-10, as mean ± std over 3 seeds.
>
> Biggest lesson: **the training recipe is worth as much as the architecture: removing augmentation costs 9 points and removing the LR schedule costs 7**.
>
> Code: https://github.com/JDevender007/Artificial_Intelligence/tree/main/p2-dl-fundamentals  #DeepLearning #PyTorch #ComputerVision
