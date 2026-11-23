"""Ask the agent one question and print its trace (the 60-second demo).

    python scripts/ask.py "What is 250 EUR in INR?"                          # Qwen2.5-7B NF4 (GPU)
    python scripts/ask.py --model Qwen/Qwen2.5-1.5B-Instruct --no-nf4 "How many employees work in Sales?"
    python scripts/ask.py --agent agent_noguard "Read the vendor onboarding policy."
"""

from __future__ import annotations

import argparse
import json
import time

from toolagent.agent import AGENTS, Episode
from toolagent.llm import HFChatLLM
from toolagent.tools import Toolbox
from toolagent.world import build_world


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no-nf4", action="store_true")
    ap.add_argument("--agent", default="agent_full", choices=list(AGENTS))
    ap.add_argument("--world", default="outputs/demo_world")
    ap.add_argument("--flaky-p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    world = build_world(a.world)
    llm = HFChatLLM(a.model, nf4=not a.no_nf4, temperature=0.0)
    ep = Episode("demo", a.question, AGENTS[a.agent], Toolbox(world, flaky_p=a.flaky_p, seed=a.seed, key="demo"))
    while not ep.done:
        t0 = time.perf_counter()
        comp = llm.generate([ep.request()], a.seed)[0]
        ep.advance(comp, time.perf_counter() - t0)
    for ev in ep.trace:
        rest = {k: v for k, v in ev.items() if k not in ("i", "step", "type")}
        print(f"[{ev['step']}] {ev['type']}: {json.dumps(rest, ensure_ascii=False)[:400]}")
    print("\nANSWER:", ep.answer)


if __name__ == "__main__":
    main()
