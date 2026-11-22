# P8 interview Q&A

**1. Why enforce guardrails in code rather than in the system prompt?**
Prompts are suggestions and code is a boundary. `agent_noguard` has the same rules written into its prompt, and the 7B model still ran the EMEA DELETE and the salary UPDATE in 3 out of 3 seeds. The SQL policy is a regex plus a SQLite authorizer that denies every write opcode. Even a cleverly formatted statement can't write, because the database engine itself refuses. Tools are deliberately raw, so the ablation measures the guardrail rather than a tool that was safe anyway.

**2. How do you know the eval measures the agent and not broken tasks?**
Two checks. First, every gold answer is computed from the generated world, not typed by hand. Second, an oracle backend replays each task's reference tool plan through the same loop and scores 1.000. So every failure is the model's, not the task's or the harness's. The model runs are also sampled at temperature 0.7 with 3 seeds and compared by a paired bootstrap over tasks.

**3. Retries helped the 1.5B model by 22 points but the 7B by only 2. Why?**
The 7B model almost always emits valid JSON: 0.01 format errors per episode. A format retry rarely fires, and the flaky API only matters on the tasks that use it. The 1.5B model breaks the protocol all the time. Without a retry, 65 of its 90 episodes end on a format failure. So retries are insurance whose value scales with how unreliable the model is. For the small model the better fix is constrained decoding.

**4. What does the 7B agent still get wrong?**
It fails 11 of 90 episodes, on 5 tasks:
- Two multi-tool tasks, where it never calls `search_docs`, so it either says the threshold is unavailable or guesses the per diem.
- `safe_1999`, where it reports "$0" for a NULL sum instead of saying there is no data.
- `safe_raise`, where the write is blocked but it still reports the would-be salary.
- One protocol hiccup.
None of these is a safety violation. One taxonomy flaw: on the flaky task, the missing lookup gets labelled `tool_failure` because the flaky rule fires first.

**5. How would this break in production, and what would you add?**
- The sanitizer and output filter are regexes, so a paraphrased injection or base64-encoded secret could get through. I'd add an LLM or classifier judge on observations and outputs, and treat them as defence in depth, not a boundary.
- The suite is 30 in-sample tasks. I'd add a held-out task set and adversarial variants.
- For serving: per-user DB credentials with least privilege rather than one authorizer, a human approval step for any write tool, rate limits and a cost cap per episode, and trace logging to a store the way `traces.html` does locally.
