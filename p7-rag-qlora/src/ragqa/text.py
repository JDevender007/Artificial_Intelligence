"""Tokenisation for BM25 and the official SQuAD answer normalisation."""

from __future__ import annotations

import re
import string

_WORD = re.compile(r"\w+", re.UNICODE)
STOPWORDS = frozenset(
    "a an the of in on at to for by with from and or is are was were be been being it its this that these those as "
    "which who whom whose what when where why how did does do has have had not no into than then there their they "
    "he she his her him we you your our".split()
)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens without stopwords (BM25 query and document side)."""
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]


def normalize_answer(s: str) -> str:
    """SQuAD v2 evaluation normalisation: lower case, drop punctuation and articles, squash whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def question_type(q: str) -> str:
    """Coarse question type from its wording, for the error analysis."""
    t = q.lower()
    for key in ("how many", "how much", "what year", "when", "who", "where", "why", "which", "how", "what"):
        if re.search(rf"\b{key}\b", t):
            return key
    return "other"
