"""Demo: answer one question with citations over the SQuAD 2.0 dev articles.

    python scripts/ask.py "When did Beyonce start becoming popular?" [--adapter outputs/squad/adapters/rag_s0]

Needs a GPU-capable environment with transformers (and peft for --adapter). Uses the retrieval setup in
``retrieval_choice.json`` of the run when it exists, else hybrid_rerank over paragraphs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragqa.chunking import make_chunks
from ragqa.config import load_config
from ragqa.data import build_dataset
from ragqa.experiment import resolve_device
from ragqa.generate import hf_backends
from ragqa.prompts import build_messages, parse_reply
from ragqa.retrieve import Retriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--config", default="configs/squad.yaml")
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    choice_path = Path(cfg.output_dir) / "retrieval_choice.json"
    choice = (json.loads(choice_path.read_text(encoding="utf-8")) if choice_path.exists()
              else {"chunker": "paragraph", "method": "hybrid_rerank"})
    r = cfg.retrieval
    b = hf_backends(cfg, resolve_device(cfg.device))
    chunks = make_chunks(build_dataset(cfg.data).dev_articles, choice["chunker"], r.window_words, r.window_overlap)
    ret = Retriever(chunks, b.embedder(), b.reranker() if choice["method"] == "hybrid_rerank" else None,
                    r.n_candidates, r.rrf_k)
    ctx = [chunks[i] for i in ret.search([a.question], choice["method"], r.k_context)[0]]
    reply = b.generator(a.adapter, False).generate([build_messages("rag", a.question, ctx)])[0]
    p = parse_reply(reply)
    print("answer:", p.answer or "(abstained: unanswerable)")
    for n in p.citations:
        if 1 <= n <= len(ctx):
            print(f"[{n}] {ctx[n - 1].article}: {ctx[n - 1].text[:300]}")
    print("\npassages given:")
    for i, c in enumerate(ctx, 1):
        print(f"  [{i}] {c.article} (chars {c.start}-{c.end})")


if __name__ == "__main__":
    main()
