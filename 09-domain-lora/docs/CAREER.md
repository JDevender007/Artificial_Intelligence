# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/09-domain-lora` (monorepo). Standalone name if split out: `domain-lora`

**Resume bullet** (fill the bracketed numbers from `reports/kaggle_t4/results.md` once the run finishes — do not guess)
- Fine-tuned Stable Diffusion 1.5 with LoRA on 1,000 CC0 Smithsonian specimen photos and evaluated it properly on a free T4: KID×1000 [X] → [Y] against 200 held-out species, CLIP score [A] → [B], species retrieval top-1 [C], with a paired bootstrap over 3 seeds and the same noise per prompt across variants. Measured memorisation instead of assuming it (max CLIP similarity to the training set [M], near-copies [N]%), and shipped a layered safety filter with its false-positive rate on benign prompts.

**LinkedIn post (draft)**
> Everyone posts the pretty grid. The questions that matter for a diffusion fine-tune are: did it actually get better, did it memorise the training set, and does the safety filter work?
>
> I trained a LoRA on 1,000 public-domain museum butterfly specimens (Smithsonian Open Access, CC0) and evaluated it three ways:
> - KID against 200 held-out species, with a **real-vs-real floor** as the scale — published FID numbers are meaningless at n=200 and I refuse to quote them
> - memorisation measured as CLIP similarity to the nearest training image, reported as both a max and the share of near-copies
> - a safety stack with the false-positive rate on benign prompts, because an unfalsifiable filter is decoration
>
> Same starting noise for every variant, 3 seeds, and the 400-step checkpoint kept next to the 800-step one so the step budget is an ablation, not a story. [repo link]

**Do not publish this draft until `reports/` exists.** The bullet has placeholders for a reason.
