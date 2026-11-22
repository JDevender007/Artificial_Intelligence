import math

import numpy as np

from ragqa.bm25 import BM25
from ragqa.chunking import Chunk, SupportIndex, paragraph_chunks, supports, window_chunks
from ragqa.data import Article, Question
from ragqa.retrieve import HashEmbedder, OverlapReranker, Retriever, rrf, top_k_rows


def test_bm25_matches_hand_computation():
    docs = ["apple banana", "apple apple cherry", "durian"]
    bm = BM25(k1=1.2, b=0.75).fit(docs)
    n, avgdl = 3, (2 + 3 + 1) / 3
    idf = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))  # "apple" is in 2 of 3 docs
    expect = [idf * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2 / avgdl)),
              idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / avgdl)), 0.0]
    np.testing.assert_allclose(bm.scores("apple"), expect)
    assert bm.search("apple", 2) == [1, 0]
    assert bm.search("unknown words", 2) == [0, 1]  # all scores zero: ties go to the lower index


def test_rrf_and_top_k():
    assert rrf([[5, 6], [6, 7]], k=60)[0] == 6  # in both lists -> highest fused score
    assert rrf([[1, 2], [2, 1]], k=60) == [1, 2]  # exact tie keeps first-seen order
    S = np.array([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0]])
    assert top_k_rows(S, 2) == [[1, 2], [0, 2]]


def _article() -> Article:
    paras = ["Alpha is a city. It was built in 1842.", "Beta is a river near Alpha."]
    return Article("Alpha", "\n\n".join(paras), (0, len(paras[0]) + 2))


def test_chunks_and_support():
    a = _article()
    pc = paragraph_chunks([a])
    assert [c.text for c in pc] == a.text.split("\n\n")
    s = a.text.index("1842")
    q = Question("q1", "When was Alpha built?", "Alpha", 0, ("1842",), (s,), False)
    assert supports(pc[0], q) and not supports(pc[1], q)
    wc = window_chunks([a], size=4, overlap=1)
    assert all(len(c.text.split()) <= 4 for c in wc)
    assert all(a.text[c.start:c.end] == c.text for c in wc)
    assert SupportIndex(wc).gold(q) == {c.cid for c in wc if c.start <= s and s + 4 <= c.end}
    imp = Question("q2", "What is the bird?", "Alpha", 0, (), (), True)
    assert SupportIndex(pc).gold(imp) == set()


def test_retriever_methods_return_k_unique_ids():
    texts = ["red apple pie", "green apple tart", "blue sky", "deep blue sea", "apple orchard red", "grey stone"]
    chunks = [Chunk(i, f"T{i}", 0, len(t), t) for i, t in enumerate(texts)]
    ret = Retriever(chunks, HashEmbedder(64), OverlapReranker(), n_candidates=5)
    for m in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        out = ret.search(["red apple"], m, 3)[0]
        assert len(out) == 3 and len(set(out)) == 3
        assert out[0] in (0, 4), m
