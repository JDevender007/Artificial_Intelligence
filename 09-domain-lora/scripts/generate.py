"""60-second demo: one prompt through the full safety stack (prompt filter -> SD 1.5 (+LoRA) -> SD checker ->
CLIP concept filter -> PNG with provenance).

    python scripts/generate.py "a museum specimen photo of a Blue Morpho, Nymphalidae family" \
        --lora outputs/main/lora/s0/step800 --seed 0 --out demo.png
"""

from __future__ import annotations

import argparse
import sys

from sdlora.config import GenConfig, ModelConfig
from sdlora.safety import SAFE_CONCEPTS, UNSAFE_CONCEPTS, check_prompt, concept_flags, save_with_provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--lora", default="", help="directory with pytorch_lora_weights.safetensors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--out", default="demo.png")
    a = ap.parse_args()

    verdict = check_prompt(a.prompt)
    if not verdict.allowed:
        print(f"BLOCKED by prompt filter: {verdict.categories}")
        sys.exit(2)

    from sdlora import gpu

    mc = ModelConfig()
    pipe = gpu.load_pipeline(mc, "cuda")
    if a.lora:
        pipe.load_lora_weights(a.lora)
    (img,), (nsfw,) = gpu.generate(pipe, [a.prompt], [a.seed], GenConfig(steps=a.steps, guidance=a.guidance))
    feat = gpu.Featurizer(mc.clip, "cuda")
    flag, mass = concept_flags(feat.image_clip([img]), feat.text_clip(UNSAFE_CONCEPTS), feat.text_clip(SAFE_CONCEPTS))
    print(f"SD safety checker flagged: {nsfw} | CLIP concept filter: unsafe mass {mass[0]:.3f}, flagged {bool(flag[0])}")
    if nsfw or flag[0]:
        print("image withheld")
        sys.exit(3)
    save_with_provenance(img, a.out, generator=mc.base, variant=a.lora or "base", prompt=a.prompt, seed=a.seed)
    print("saved", a.out, "(PNG text chunk ai_generated=true)")


if __name__ == "__main__":
    main()
