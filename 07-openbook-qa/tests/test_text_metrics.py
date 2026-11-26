import numpy as np
import pytest

from ragqa.chunking import Chunk
from ragqa.metrics import contains, context_precision, mrr_at_k, paired_bootstrap, recall_at_k, squad_em, squad_f1
from ragqa.prompts import ABSTAIN, build_messages, parse_reply, target_text
from ragqa.text import normalize_answer, question_type


def test_normalize_and_squad_scores():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert squad_em("the Eiffel tower", ["Eiffel Tower"]) == 1.0
    assert squad_f1("Eiffel", ["Eiffel Tower"]) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert squad_f1("", []) == 1.0 and squad_em("Paris", []) == 0.0  # unanswerable: only empty is right
    assert squad_f1("", ["Paris"]) == 0.0


def test_retrieval_metrics():
    assert recall_at_k([3, 1, 2], {2}, 2) == 0.0 and recall_at_k([3, 1, 2], {2}, 3) == 1.0
    assert mrr_at_k([3, 1, 2], {1}) == 0.5
    assert context_precision([7, 8, 9], {7, 9}) == pytest.approx((1 + 2 / 3) / 2)  # relevant at ranks 1 and 3
    assert context_precision([1, 2], set()) == 0.0
    assert contains("It opened in 1889.", "1889") and not contains("in 18890", "1889")


def test_paired_bootstrap_sign():
    a, b = np.zeros(200), np.ones(200)
    d = paired_bootstrap(a, b, n=200)
    assert d["diff"] == 1.0 and d["lo"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:10])


def test_parse_reply():
    p = parse_reply("Answer: 1842 [2].\nBecause ...")
    assert p.answer == "1842" and p.citations == (2,) and not p.abstained
    assert parse_reply(ABSTAIN).abstained
    assert parse_reply("The passages do not contain the answer.").abstained
    assert parse_reply("").abstained


def test_prompts_and_targets():
    ch = [Chunk(0, "Alpha", 0, 10, "Alpha text")]
    m = build_messages("rag", "Q?", ch)
    assert m[0]["role"] == "system" and "[1] Alpha: Alpha text" in m[1]["content"]
    assert build_messages("base", "Q?") == [{"role": "user", "content": "Q?"}]
    with pytest.raises(ValueError):
        build_messages("rag", "Q?")
    assert target_text("1842", 2) == "1842 [2]" and target_text(None) == ABSTAIN


def test_question_type():
    assert question_type("How many people live there?") == "how many"
    assert question_type("In what year was it built?") == "what year"
    assert question_type("Name the river.") == "other"
