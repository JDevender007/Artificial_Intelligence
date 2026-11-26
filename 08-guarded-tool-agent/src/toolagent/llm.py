"""LLM backends behind one batched interface.

* ``HFChatLLM``: a Hugging Face chat model (Qwen2.5-Instruct), fp16 or 4-bit NF4, left-padded batches, nucleus
  sampling seeded per call so a (seed, config) run is reproducible on the same hardware and library versions.
* ``ScriptedLLM``: a plain function of the messages, used by the tests and by the ``oracle`` backend, which replays
  each task's reference plan. The oracle proves every task is solvable with the tools; it is not a baseline.

Only ``HFChatLLM`` imports torch / transformers, and only when it is constructed.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

Messages = list[dict[str, str]]


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLM(Protocol):
    name: str

    def generate(self, batch: list[Messages], seed: int) -> list[Completion]: ...


def _words(messages: Messages) -> int:
    return sum(len(m["content"].split()) for m in messages)


class ScriptedLLM:
    """Deterministic policy ``fn(messages) -> reply``; token counts are whitespace word counts."""

    def __init__(self, fn: Callable[[Messages], str], name: str = "scripted"):
        self.fn, self.name = fn, name

    def generate(self, batch: list[Messages], seed: int) -> list[Completion]:
        out = []
        for m in batch:
            text = self.fn(m)
            out.append(Completion(text, _words(m), len(text.split())))
        return out


def _dtype_kw(torch_dtype: Any) -> dict[str, Any]:
    """transformers >= 4.56 takes ``dtype=``; older versions only ``torch_dtype=``."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": torch_dtype} if (major, minor) >= (4, 56) else {"torch_dtype": torch_dtype}


class HFChatLLM:
    """Batched chat generation. ``temperature=0`` means greedy decoding."""

    def __init__(self, model: str, device: str = "cuda", dtype: str = "float16", nf4: bool = False,
                 max_new_tokens: int = 192, temperature: float = 0.7, top_p: float = 0.9, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.name = torch, model
        self.max_new, self.temperature, self.top_p, self.bs = max_new_tokens, temperature, top_p, batch_size
        dt = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if nf4:
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=dt)
            m = AutoModelForCausalLM.from_pretrained(model, quantization_config=q, device_map={"": 0})
        else:
            m = AutoModelForCausalLM.from_pretrained(model, **_dtype_kw(dt)).to(device)
        self.model = m.eval()
        self._calls = 0

    def generate(self, batch: list[Messages], seed: int) -> list[Completion]:
        torch = self.torch
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        lengths = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = np.argsort(lengths, kind="stable")
        out: list[Completion | None] = [None] * len(prompts)
        sample = self.temperature > 0
        kw: dict[str, Any] = ({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p, "top_k": 0}
                              if sample else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
        with torch.inference_mode():
            for i in range(0, len(order), self.bs):
                idx = order[i:i + self.bs]
                torch.manual_seed(seed * 1_000_003 + self._calls)
                self._calls += 1
                enc = self.tok([prompts[j] for j in idx], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.model.device)
                gen = self.model.generate(**enc, max_new_tokens=self.max_new, pad_token_id=self.tok.pad_token_id, **kw)
                new = gen[:, enc["input_ids"].shape[1]:]
                texts = self.tok.batch_decode(new, skip_special_tokens=True)
                for j, t, row in zip(idx, texts, new, strict=True):
                    n_new = int((row != self.tok.pad_token_id).sum())
                    out[int(j)] = Completion(t, lengths[int(j)], n_new)
        return [c for c in out if c is not None]

    def reset(self) -> None:
        """Restart the per-run call counter so each (config, seed) run draws the same sampling seeds."""
        self._calls = 0

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
