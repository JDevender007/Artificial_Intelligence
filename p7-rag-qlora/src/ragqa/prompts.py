"""Chat prompts for each variant, the fine-tuning targets, and parsing of the model's reply."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragqa.chunking import Chunk

ABSTAIN = "unanswerable"
INSTR_CLOSED = (
    "Answer the question with the shortest possible phrase (a name, number, date or short noun phrase), "
    f"not a sentence. If you do not know the answer or the question has no answer, reply exactly: {ABSTAIN}"
)
INSTR_RAG = (
    "Answer the question using only the numbered passages. Reply with the shortest possible phrase copied from "
    "the passages, followed by the number of the passage that supports it in square brackets, for example: "
    f"1842 [2]. If the passages do not contain the answer, reply exactly: {ABSTAIN}"
)
# Phrases counted as abstaining when a model ignores the exact keyword (the unprompted base model never sees it).
_ABSTAIN_PHRASES = ("unanswerable", "i don't know", "i do not know", "not mentioned", "cannot be answered",
                    "no answer", "not provided", "does not mention", "do not contain", "doesn't mention",
                    "not contain", "no information")
_CITE = re.compile(r"\[(\d+)\]")

Messages = list[dict[str, str]]


def passages_block(chunks: list[Chunk]) -> str:
    return "\n".join(f"[{i}] {c.article}: {' '.join(c.text.split())}" for i, c in enumerate(chunks, start=1))


def build_messages(variant: str, question: str, chunks: list[Chunk] | None = None) -> Messages:
    """``base``: the bare question; closed-book prompted: instruction; rag: instruction + numbered passages."""
    if variant == "base":
        return [{"role": "user", "content": question}]
    if variant in ("prompted", "ft_closed"):
        return [{"role": "system", "content": INSTR_CLOSED}, {"role": "user", "content": f"Question: {question}"}]
    if variant in ("rag", "rag_nf4", "rag_ft"):
        if chunks is None:
            raise ValueError("rag variants need passages")
        return [{"role": "system", "content": INSTR_RAG},
                {"role": "user", "content": f"Passages:\n{passages_block(chunks)}\n\nQuestion: {question}"}]
    raise ValueError(variant)


def target_text(answer: str | None, cite: int | None = None) -> str:
    """Fine-tuning target: ``answer [cite]``, the bare answer (closed book), or the abstain keyword."""
    if answer is None:
        return ABSTAIN
    return answer if cite is None else f"{answer} [{cite}]"


@dataclass(frozen=True)
class Parsed:
    answer: str  # "" when the model abstained
    citations: tuple[int, ...]  # 1-based passage numbers as written by the model (may be out of range)
    abstained: bool


def parse_reply(text: str) -> Parsed:
    """Keep the first non-empty line, read ``[n]`` citations, strip them and an ``Answer:`` prefix."""
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^(final\s+)?answer\s*[:\-]\s*", "", line, flags=re.I)
    cites = tuple(int(x) for x in _CITE.findall(line))
    ans = _CITE.sub("", line).strip().strip(" .;,:\"'").strip()
    low = ans.lower()
    if not ans or any(p in low for p in _ABSTAIN_PHRASES):
        return Parsed("", cites, True)
    return Parsed(ans, cites, False)
