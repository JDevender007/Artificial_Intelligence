# P9 60-second demo script

1. **(0-8 s)** "A Stable Diffusion 1.5 LoRA trained on 1,000 public-domain museum butterfly specimens — and evaluated the three ways diffusion projects usually skip."

2. **(8-20 s)** Show `outputs/main/grids/` next to the real held-out images. "Same prompt, same starting noise, three variants: the base model, the 400-step LoRA and the 800-step LoRA. Paired, so the comparison isn't two independent samples."

3. **(20-35 s)** Run `python scripts\generate.py "a museum specimen photo of a Morpho butterfly, Nymphalidae family" --adapter outputs\main\lora\s0` and open the PNG metadata. "Every image carries provenance — base model, adapter, prompt, `ai_generated=true` — written into the file itself."

4. **(35-48 s)** Open `reports/kaggle_t4/results.md`. "KID against 200 held-out species, next to the real-vs-real floor. FID is reported too, but at n=200 it's biased, so the floor is the only honest scale. CLIP score, species retrieval top-1 and top-5, and memorisation: max CLIP similarity to the training set and the share of near-copies."

5. **(48-60 s)** Open `reports/kaggle_t4/safety.md`. "Three layers: a prompt filter on raw and de-obfuscated text, a CLIP concept filter plus SD's safety checker, and provenance. The false-positive rate on benign prompts is here too, because a filter you can't falsify isn't a safety feature." End on the green CI badge.
