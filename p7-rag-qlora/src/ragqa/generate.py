"""Generator protocol, a deterministic fake for tests, and the backend bundle the experiment runs with."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ragqa.prompts import ABSTAIN, Messages
from ragqa.retrieve import Embedder, HashEmbedder, OverlapReranker, Reranker
from ragqa.text import tokenize


class Generator(Protocol):
    def generate(self, batch: list[Messages]) -> list[str]:
        """Greedy reply text for each conversation."""
        ...

    def close(self) -> None:
        """Free the model (GPU memory)."""
        ...


class RuleGenerator:
    """Test stand-in. With passages: the first word of the passage with most question overlap that is not in
    the question, cited. Without passages: abstains."""

    _P = re.compile(r"^\[(\d+)\] [^:]*: (.*)$")

    def generate(self, batch: list[Messages]) -> list[str]:
        out = []
        for msgs in batch:
            user = msgs[-1]["content"]
            q = user.rsplit("Question:", 1)[-1]
            qt = set(tokenize(q))
            best, best_n, best_ov = None, 0, -1
            for line in user.splitlines():
                m = self._P.match(line)
                if m:
                    ov = len(qt & set(tokenize(m.group(2))))
                    if ov > best_ov:
                        best, best_n, best_ov = m.group(2), int(m.group(1)), ov
            if best is None:
                out.append(ABSTAIN)
                continue
            words = [w.strip(".,") for w in best.split() if w.strip(".,").lower() not in qt and w[:1].isupper()]
            out.append(f"{words[0]} [{best_n}]" if words else ABSTAIN)
        return out

    def close(self) -> None:
        pass


SFTExample = dict[str, Any]  # {"messages": Messages, "target": str}


@dataclass
class Backends:
    """Factories for the heavy components, so tests can swap in fakes."""

    embedder: Callable[[], Embedder]
    reranker: Callable[[], Reranker]
    generator: Callable[[str | None, bool], Generator]  # (adapter dir or None, load base in 4-bit NF4)
    trainer: Callable[[list[SFTExample], int, Path], dict[str, Any]]  # (examples, seed, adapter dir) -> log


def fake_trainer(examples: list[SFTExample], seed: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_config.json").write_text(json.dumps({"fake": True, "seed": seed}), encoding="utf-8")
    return {"seed": seed, "n_examples": len(examples), "steps": 1, "losses": [1.0], "train_s": 0.0,
            "skipped_too_long": 0}


def fake_backends() -> Backends:
    return Backends(embedder=HashEmbedder, reranker=OverlapReranker,
                    generator=lambda adapter=None, nf4=False: RuleGenerator(), trainer=fake_trainer)


def hf_backends(cfg: Any, device: str) -> Backends:
    """Real backends (Hugging Face). Imported lazily so the package and its tests need no torch/transformers."""
    from ragqa import hf

    r, g = cfg.retrieval, cfg.gen
    return Backends(
        embedder=lambda: hf.HFEmbedder(r.embedder, device, r.batch_size, r.max_len, r.query_prefix),
        reranker=lambda: hf.HFReranker(r.reranker, device, r.batch_size, r.max_len),
        generator=lambda adapter=None, nf4=False: hf.HFGenerator(g.model, device, g.dtype, g.batch_size,
                                                                 g.max_new_tokens, adapter=adapter, nf4=nf4),
        trainer=lambda ex, seed, out: hf.train_qlora(g.model, ex, cfg.ft, seed, out, device, g.dtype),
    )
