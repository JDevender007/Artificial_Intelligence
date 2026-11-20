# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/p2-dl-fundamentals` (monorepo). Standalone name if split out: `pytorch-from-scratch-cnn`

## Resume bullet (fill in from `reports/*/results.md` once the runs finish)

- Implemented an MLP and CNN from scratch in PyTorch (custom linear layer, hand-written training loop with fp16 mixed precision, OneCycle LR, early stopping, resumable checkpoints, TensorBoard). Reached **[CNN acc ± s]** CIFAR-10 test accuracy with 1.15M parameters vs **[baseline acc]** for softmax regression and **[ResNet acc]** for a fine-tuned ResNet-18 (3 seeds).
- Ran ablations showing augmentation adds **[Δ]** points and mixed precision gives **[speed-up]×** throughput at equal accuracy. 34 pytest tests run on CPU in GitHub Actions.

## LinkedIn post draft

> Project 2/11 of my AI-engineering portfolio: **deep learning without the training wheels**.
>
> I wrote the linear layer (`x @ W.T + b`) and the whole training loop myself in PyTorch: mixed precision, OneCycle LR, early stopping, checkpoint resume, TensorBoard. Then I compared:
> - softmax regression baseline: **[ ]**
> - my MLP: **[ ]**
> - my 1.15M-param CNN: **[ ]**
> - pretrained ResNet-18 fine-tune: **[ ]**
>
> All on CIFAR-10, as mean ± std over 3 seeds.
>
> Biggest lesson: **[from the ablations / confusion matrix]**.
>
> Code: https://github.com/JDevender007/Artificial_Intelligence/tree/main/p2-dl-fundamentals  #DeepLearning #PyTorch #ComputerVision
