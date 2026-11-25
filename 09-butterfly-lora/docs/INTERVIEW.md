# P9 interview Q&A

**1. Why KID instead of FID as the headline metric?**
Because the reference set is 200 images. FID's estimate is biased upward at that sample size and cannot be compared with published FIDs on 50k-image sets; KID is unbiased at small n. FID is still reported next to it, and both are read against a **real-vs-real floor** — held-out real images scored against real training images — which is the only meaningful scale here.

**2. How do you know the model didn't memorise the training images?**
It is measured, not assumed: every generated image is embedded with CLIP ViT-B/16, its maximum cosine similarity to any training image is recorded, and the share of images above 0.95 (near-copies) is reported. The real held-out images provide the natural level for comparison. A CLIP-similarity check is a proxy — a pixel-space nearest-neighbour test is the stronger version and is listed as future work.

**3. Why freeze the VAE and text encoder, and why keep the LoRA weights in fp32?**
1,000 images cannot justify adapting a text encoder, and caching VAE latents plus caption embeddings is what makes 800 steps fit a free T4. LoRA parameters stay fp32 under autocast: at lr 1e-4 that is stable, while fp16 LoRA weights diverge. Gradient checkpointing plus grad-norm clipping keeps the frozen UNet in fp16 without OOM.

**4. The comparison is "paired" — what exactly is paired, and why does it matter?**
The noise seed for prompt *i* under run seed *s* is `s * 100_000 + i` and is shared by every variant, so base and both LoRA checkpoints start from identical latents for the same caption. Without that, differences between variants include sampling noise, and with 200 images that noise is large enough to reorder the results.

**5. What do the three safety layers actually guarantee?**
They reduce, and they are measured. The prompt filter catches obvious categories on both raw and de-obfuscated text; the CLIP concept filter plus SD's bundled checker suppress unsafe images; provenance makes every output traceable to a model and prompt. None of them is a guarantee — CLIP-based concept filtering has documented gaps — which is why the false-positive rate on benign prompts is reported alongside the detection behaviour instead of a single "safety score".
