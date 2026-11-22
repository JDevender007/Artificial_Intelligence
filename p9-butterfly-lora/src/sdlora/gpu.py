"""GPU side: SD 1.5 pipeline, LoRA training, generation and feature extraction (torch, diffusers, peft, transformers).

Imported only by :mod:`sdlora.experiment` at run time, so the unit tests and CI never need torch.

Training follows the diffusers ``train_text_to_image_lora`` recipe (LoRA on the UNet attention projections, frozen
VAE and text encoder, epsilon-prediction MSE), simplified for a 16 GB T4: VAE latents (posterior mean) and caption
embeddings are computed once and cached, the frozen UNet stays in fp16, the LoRA parameters are fp32 under autocast
with a GradScaler, and gradient checkpointing is on.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sdlora.config import GenConfig, ModelConfig, TrainConfig
from sdlora.utils import get_logger

log = get_logger(__name__)
LORA_TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def load_pipeline(mc: ModelConfig, device: str, sd_checker: bool = True):
    """StableDiffusionPipeline with DPM-Solver++ (25 steps is enough) and the bundled safety checker."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    kw: dict[str, Any] = {"torch_dtype": _dtype(mc.dtype)}
    if mc.revision:
        kw["revision"] = mc.revision
    if not sd_checker:
        kw.update(safety_checker=None, requires_safety_checker=False)
    pipe = StableDiffusionPipeline.from_pretrained(mc.base, **kw)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@torch.no_grad()
def encode_dataset(pipe, images: list[Image.Image], captions: list[str], batch_size: int = 16
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached VAE latents (posterior mean x scaling factor) and CLIP-text caption embeddings, both fp16 on CPU."""
    dev, dt = pipe.device, pipe.vae.dtype
    lats, embs = [], []
    for i in range(0, len(images), batch_size):
        x = torch.stack([torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1).permute(2, 0, 1)
                         for im in images[i:i + batch_size]]).to(dev, dt)
        lats.append((pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor).half().cpu())
        ids = pipe.tokenizer(captions[i:i + batch_size], padding="max_length", truncation=True,
                             max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(dev)
        embs.append(pipe.text_encoder(ids)[0].half().cpu())
    return torch.cat(lats), torch.cat(embs)


def train_lora(mc: ModelConfig, tc: TrainConfig, latents: torch.Tensor, embeds: torch.Tensor, seed: int,
               out_dir: Path, device: str, on_log: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Train one LoRA from a fresh UNet; save ``out_dir/step<k>`` at every checkpoint. Returns timing and losses."""
    from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    torch.manual_seed(seed)
    kw = {"revision": mc.revision} if mc.revision else {}
    unet = UNet2DConditionModel.from_pretrained(mc.base, subfolder="unet", torch_dtype=_dtype(mc.dtype), **kw)
    noise_sched = DDPMScheduler.from_pretrained(mc.base, subfolder="scheduler", **kw)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=tc.rank, lora_alpha=tc.alpha, init_lora_weights="gaussian",
                                target_modules=LORA_TARGETS))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    if tc.grad_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.to(device).train()
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)

    def lr_at(step: int) -> float:  # linear warm-up, then cosine to 0
        if step < tc.warmup:
            return (step + 1) / tc.warmup
        return 0.5 * (1 + math.cos(math.pi * (step - tc.warmup) / max(tc.steps - tc.warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = mc.dtype == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    n_t = noise_sched.config.num_train_timesteps
    losses: list[float] = []
    t0 = time.time()
    for step in range(1, tc.steps + 1):
        idx = torch.randint(len(latents), (tc.batch_size,), generator=g)
        lat = latents[idx].to(device, torch.float32)
        emb = embeds[idx].to(device)
        noise = torch.randn(lat.shape, generator=g).to(device)
        t = torch.randint(0, n_t, (tc.batch_size,), generator=g).to(device)
        noisy = noise_sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred = unet(noisy, t, encoder_hidden_states=emb).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(loss.item())
        if on_log and (step % 50 == 0 or step == tc.steps):
            on_log({"seed": seed, "step": step, "loss_avg50": float(np.mean(losses[-50:])),
                    "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
        if step in tc.checkpoints:
            layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            StableDiffusionPipeline.save_lora_weights(save_directory=str(out_dir / f"step{step}"),
                                                      unet_lora_layers=layers, safe_serialization=True)
    train_s = time.time() - t0
    del unet, opt
    torch.cuda.empty_cache()
    return {"train_s": train_s, "s_per_step": train_s / tc.steps, "lora_params": n_params,
            "final_loss_avg50": float(np.mean(losses[-50:]))}


@torch.no_grad()
def generate(pipe, prompts: list[str], noise_seeds: list[int], gc: GenConfig) -> tuple[list[Image.Image], list[bool]]:
    """One image per prompt; each prompt has its own generator seed, so base and LoRA share initial noise."""
    images: list[Image.Image] = []
    flags: list[bool] = []
    for i in range(0, len(prompts), gc.batch_size):
        gens = [torch.Generator(pipe.device).manual_seed(s) for s in noise_seeds[i:i + gc.batch_size]]
        out = pipe(prompts[i:i + gc.batch_size], num_inference_steps=gc.steps, guidance_scale=gc.guidance,
                   generator=gens)
        images += out.images
        nsfw = getattr(out, "nsfw_content_detected", None)
        flags += [bool(x) for x in nsfw] if nsfw is not None else [False] * len(out.images)
    return images, flags


class Featurizer:
    """CLIP image/text embeddings (L2-normalised) and Inception-v3 pool features for FID/KID."""

    def __init__(self, clip_name: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_name, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(clip_name)
        self._inception = None

    @torch.no_grad()
    def image_clip(self, images: list[Image.Image], bs: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), bs):
            px = self.proc(images=images[i:i + bs], return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
            # vision tower + projection directly: get_image_features' return type changed across transformers releases
            e = self.clip.visual_projection(self.clip.vision_model(pixel_values=px).pooler_output)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def text_clip(self, texts: list[str], bs: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            tok = self.proc(text=texts[i:i + bs], padding=True, truncation=True, return_tensors="pt").to(self.device)
            h = self.clip.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output
            e = self.clip.text_projection(h)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def inception(self, images: list[Image.Image], bs: int = 50) -> np.ndarray:
        """2048-d pool features from torchmetrics' FID Inception (the TF-FID weights, via torch-fidelity)."""
        if self._inception is None:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self._inception = FrechetInceptionDistance(feature=2048).inception.to(self.device).eval()
        out = []
        for i in range(0, len(images), bs):
            x = torch.stack([torch.from_numpy(np.asarray(im.convert("RGB"), np.uint8).copy()).permute(2, 0, 1)
                             for im in images[i:i + bs]]).to(self.device)
            out.append(self._inception(x).double().cpu().numpy().reshape(len(x), -1))
        return np.concatenate(out)
