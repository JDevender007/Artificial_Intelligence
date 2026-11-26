"""Chunkers over article text, and the gold-support relation between chunks and questions."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from ragqa.data import PARA_SEP, Article, Question

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    cid: int
    article: str
    start: int  # article-level character span [start, end)
    end: int
    text: str

    @property
    def index_text(self) -> str:
        """What the retrievers see: the article title gives each chunk its topic."""
        return f"{self.article}\n{self.text}"


def paragraph_chunks(articles: list[Article]) -> list[Chunk]:
    """One chunk per SQuAD paragraph."""
    out = []
    for a in articles:
        for i, s in enumerate(a.para_offsets):
            e = a.para_offsets[i + 1] - len(PARA_SEP) if i + 1 < len(a.para_offsets) else len(a.text)
            out.append(Chunk(len(out), a.title, s, e, a.text[s:e]))
    return out


def window_chunks(articles: list[Article], size: int, overlap: int) -> list[Chunk]:
    """Fixed windows of ``size`` whitespace tokens, stepping ``size - overlap``; they may cross paragraphs."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    out = []
    step = size - overlap
    for a in articles:
        toks = [(m.start(), m.end()) for m in _TOKEN.finditer(a.text)]
        i = 0
        while i < len(toks):
            j = min(i + size, len(toks))
            s, e = toks[i][0], toks[j - 1][1]
            out.append(Chunk(len(out), a.title, s, e, a.text[s:e]))
            if j == len(toks):
                break
            i += step
    return out


def make_chunks(articles: list[Article], chunker: str, window_words: int = 100, window_overlap: int = 25) -> list[Chunk]:
    if chunker == "paragraph":
        return paragraph_chunks(articles)
    if chunker == "window":
        return window_chunks(articles, window_words, window_overlap)
    raise ValueError(chunker)


def supports(chunk: Chunk, q: Question) -> bool:
    """True when the chunk contains at least one whole gold answer span of the question."""
    return chunk.article == q.article and any(
        chunk.start <= s and s + len(a) <= chunk.end for a, s in zip(q.answers, q.starts, strict=True))


class SupportIndex:
    """Gold supporting chunk ids per question (empty for unanswerable questions)."""

    def __init__(self, chunks: list[Chunk]):
        self.by_article: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            self.by_article[c.article].append(c)

    def gold(self, q: Question) -> set[int]:
        if q.is_impossible:
            return set()
        return {c.cid for c in self.by_article.get(q.article, []) if supports(c, q)}
