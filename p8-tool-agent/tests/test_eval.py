from __future__ import annotations

import re

import pytest

from toolagent.agent import AGENTS
from toolagent.config import load_config
from toolagent.evaluate import run_episodes, run_experiment, score_episode
from toolagent.llm import ScriptedLLM
from toolagent.tasks import CATEGORIES, build_tasks, numbers_in, oracle_policy
from toolagent.utils import read_jsonl
from toolagent.viewer import write_viewer
from toolagent.world import build_world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("w") / "world", seed=0)


def test_suite_shape(world):
    tasks = build_tasks(world)
    assert len(tasks) == 30 and len({t.id for t in tasks}) == 30 and len({t.prompt for t in tasks}) == 30
    counts = {c: sum(t.category == c for t in tasks) for c in CATEGORIES}
    assert counts == {"lookup": 6, "sql": 6, "calc": 5, "multi": 7, "safety": 6}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_solves_everything_with_full_agent(world, seed):
    tasks = build_tasks(world)
    eps = run_episodes(ScriptedLLM(oracle_policy(tasks)), AGENTS["agent_full"], tasks, world, seed, flaky_p=0.5)
    by_id = {t.id: t for t in tasks}
    scores = {e.task_id: score_episode(by_id[e.task_id], e) for e in eps}
    assert all(s["success"] for s in scores.values()), {k: s["failure"] for k, s in scores.items()}


def test_gold_numbers_come_from_tool_outputs(world):
    """The last observation of each numeric plan contains the gold: the plans really compute the answer."""
    tasks = build_tasks(world)
    eps = run_episodes(ScriptedLLM(oracle_policy(tasks)), AGENTS["agent_full"], tasks, world, 0, flaky_p=0.0)
    for t, e in zip(tasks, eps, strict=True):
        if t.check != "number":
            continue
        obs = [ev["content"] for ev in e.trace if ev["type"] == "observation"]
        assert obs, t.id
        nums = numbers_in(re.sub(r"\b\d+ (EUR|USD|km)\b", "", obs[-1])) or numbers_in(obs[-1])
        assert any(abs(x - t.gold) <= t.tol for x in nums), (t.id, t.gold, obs[-1])


def test_injection_is_sanitized_only_with_guardrails(world):
    tasks = [t for t in build_tasks(world) if t.id == "safe_injection"]
    llm = ScriptedLLM(oracle_policy(tasks))
    full = run_episodes(llm, AGENTS["agent_full"], tasks, world, 0, 0.0)[0]
    raw = run_episodes(llm, AGENTS["agent_noguard"], tasks, world, 0, 0.0)[0]
    assert full.stats.injections_removed >= 1 and "ignore all previous" not in full.messages[-1]["content"].lower()
    assert raw.stats.injections_removed == 0 and "ignore all previous" in raw.messages[-1]["content"].lower()


def test_end_to_end_reports_and_resume(tmp_path):
    out = tmp_path / "run"
    cfg = load_config("configs/test.yaml", [f"output_dir={out.as_posix()}"])
    res = run_experiment(cfg, check=True)
    rows = read_jsonl(out / "episodes.jsonl")
    assert len(rows) == 30 * len(AGENTS) * 2
    full = [r for r in res["aggregate"] if r["run"] == "oracle / agent_full"][0]
    assert full["success_mean"] == 1.0 and full["violations_mean"] == 0
    for f in ["results.md", "categories.md", "bootstrap.md", "failures.md", "results.json"]:
        assert (out / f).stat().st_size > 0
    run_experiment(cfg)  # resume: nothing re-run
    assert len(read_jsonl(out / "episodes.jsonl")) == len(rows)
    n = write_viewer(out / "traces", out / "traces.html")
    html = (out / "traces.html").read_text(encoding="utf-8")
    assert n == len(rows) and "<\\/script>" not in html.split('id="data"')[0]
    assert html.count("</script>") == 2
