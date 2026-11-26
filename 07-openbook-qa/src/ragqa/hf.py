"""Hugging Face backends: bge embedder, bge cross-encoder reranker, Qwen generator and the QLoRA trainer.

Only this module imports torch / transformers / peft / bitsandbytes, and only when it is used.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ragqa.prompts import Messages
from ragqa.utils import get_logger, set_seed, write_json

log = get_logger()


def _torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _dtype_kw(dt) -> dict[str, Any]:
    """transformers >= 4.56 takes ``dtype=``; older versions only ``torch_dtype=``."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dt} if (major, minor) >= (4, 56) else {"torch_dtype": dt}


def free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class HFEmbedder:
    """bge-style bi-encoder: CLS pooling, L2 normalisation, an instruction prefix on queries only."""

    def __init__(self, name: str, device: str, batch_size: int = 64, max_len: int = 512, query_prefix: str = ""):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device, self.bs, self.max_len, self.prefix = torch, device, batch_size, max_len, query_prefix
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        if device.startswith("cuda"):
            self.model.half()

    def encode(self, texts: list[str], kind: Literal["query", "doc"]) -> np.ndarray:
        torch = self.torch
        if kind == "query":
            texts = [self.prefix + t for t in texts]
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.bs):
                enc = self.tok(texts[i:i + self.bs], padding=True, truncation=True, max_length=self.max_len,
                               return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state[:, 0].float()
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


class HFReranker:
    """Cross-encoder with a single relevance logit (e.g. BAAI/bge-reranker-base)."""

    def __init__(self, name: str, device: str, batch_size: int = 64, max_len: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device, self.bs, self.max_len = torch, device, batch_size, max_len
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        if device.startswith("cuda"):
            self.model.half()

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(pairs), self.bs):
                q, p = zip(*pairs[i:i + self.bs], strict=True)
                enc = self.tok(list(q), list(p), padding=True, truncation="only_second", max_length=self.max_len,
                               return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def _bnb_config(dtype: str):
    import torch
    from transformers import BitsAndBytesConfig

    compute = torch.float16 if dtype == "float32" else _torch_dtype(dtype)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=compute)


def _tokenizer(name: str, side: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, padding_side=side)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class HFGenerator:
    """Greedy decoding with the chat template, batched with left padding, sorted by length."""

    def __init__(self, name: str, device: str, dtype: str = "float16", batch_size: int = 16, max_new_tokens: int = 32,
                 adapter: str | None = None, nf4: bool = False):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch, self.device, self.bs, self.max_new = torch, device, batch_size, max_new_tokens
        self.tok = _tokenizer(name, "left")
        if nf4:
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=_bnb_config(dtype),
                                                         device_map={"": 0})
        else:
            model = AutoModelForCausalLM.from_pretrained(name, **_dtype_kw(_torch_dtype(dtype))).to(device)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            if not nf4:
                model = model.merge_and_unload()
        gc_ = model.generation_config
        for k in ("temperature", "top_p", "top_k"):  # pure greedy: drop the sampling defaults of the checkpoint
            if hasattr(gc_, k):
                setattr(gc_, k, None)
        self.model = model.eval()

    def generate(self, batch: list[Messages]) -> list[str]:
        torch = self.torch
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        lengths = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = np.argsort(lengths)
        out: list[str] = [""] * len(prompts)
        with torch.inference_mode():
            for i in range(0, len(order), self.bs):
                idx = order[i:i + self.bs]
                enc = self.tok([prompts[j] for j in idx], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.model.device)
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                          repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
                texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j, t in zip(idx, texts, strict=True):
                    out[int(j)] = t
        return out

    def close(self) -> None:
        del self.model
        free_cuda()


def train_qlora(name: str, examples: list[dict[str, Any]], ft: Any, seed: int, out_dir: Path, device: str,
                dtype: str = "float16") -> dict[str, Any]:
    """QLoRA: 4-bit NF4 frozen base + LoRA adapters; loss only on the target tokens. Saves the adapter."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    set_seed(seed)
    tok = _tokenizer(name, "right")
    seqs, skipped = [], 0
    for ex in examples:
        p = tok(tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)["input_ids"]
        t = tok(ex["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p) + len(t) > ft.max_seq_len:
            skipped += 1
            continue
        seqs.append((p + t, [-100] * len(p) + t))
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=_bnb_config(dtype), device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=ft.lora_r, lora_alpha=ft.lora_alpha, lora_dropout=ft.lora_dropout,
                      target_modules=list(ft.target_modules), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:  # GradScaler needs fp32 master weights for the trainable adapter
        p.data = p.data.float()
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())

    rng = np.random.default_rng(seed)
    per_step = ft.batch_size * ft.grad_accum
    total = math.ceil(len(seqs) / per_step) * ft.epochs
    if ft.max_steps:
        total = min(total, ft.max_steps)
    warm = max(1, int(ft.warmup_ratio * total))
    opt = torch.optim.AdamW(params, lr=ft.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    use_scaler = dtype == "float16"
    amp_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    pad = tok.pad_token_id

    def batches():
        for _ in range(ft.epochs):
            perm = rng.permutation(len(seqs))
            # group by length within blocks of 64 to cut padding, then shuffle the batch order
            blocks = [sorted(perm[i:i + 64], key=lambda j: len(seqs[j][0])) for i in range(0, len(perm), 64)]
            bl = [b[i:i + ft.batch_size] for b in blocks for i in range(0, len(b), ft.batch_size)]
            for k in rng.permutation(len(bl)):
                yield bl[k]

    model.train()
    losses, step, micro, acc, tokens = [], 0, 0, 0.0, 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for b in batches():
        L = max(len(seqs[j][0]) for j in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long)
        lab = torch.full((len(b), L), -100, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b):
            x, y = seqs[j]
            ids[r, :len(x)] = torch.tensor(x)
            lab[r, :len(y)] = torch.tensor(y)
            att[r, :len(x)] = 1
            tokens += len(x)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
        scaler.scale(loss / ft.grad_accum).backward()
        acc += float(loss.detach())
        micro += 1
        if micro % ft.grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(acc / ft.grad_accum)
            acc, step = 0.0, step + 1
            if step % 20 == 0 or step == total:
                log.info("qlora seed %d step %d/%d loss %.4f (%.0fs)", seed, step, total, losses[-1], time.time() - t0)
            if step >= total:
                break
    train_s = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    info = {"seed": seed, "n_examples": len(seqs), "skipped_too_long": skipped, "steps": step, "losses": losses,
            "train_s": train_s, "tokens": tokens, "trainable_params": n_trainable, "total_params": n_total,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    write_json(info, out_dir / "train_info.json")
    del model, opt
    free_cuda()
    return info
