# P8 career assets

**Repo folder:** `08-tool-agent` (suggested standalone name: `tool-agent-guardrails-eval`)

**Resume bullet**
- Built a JSON-protocol tool-using agent: 8 tools, SQL/file/output guardrails enforced in code, and format and tool retries. Evaluated it on a 30-task suite (2 Qwen2.5 models Ã— 4 ablations Ã— 3 seeds) on a free T4. Tools raised the 7B model from 0.244 to 0.878 success (bootstrap CI +0.46 to +0.80). Guardrails cut unsafe actions from 7 to 0 at no cost in task success. Retries lifted the 1.5B model by +0.22.

**LinkedIn post (draft)**
> I gave a 7B open model a SQL tool and asked it to delete all EMEA orders. It did, and then reported "successfully deleted".
>
> So I built the guardrails in code instead of the prompt, then measured them. 30 tasks, 2 Qwen2.5 models, 4 agent variants, 3 seeds, all on a free Kaggle T4:
> - Tools take Qwen2.5-7B from 24% to 88% success. The 1.5B model *with* tools beats the 7B without them.
> - With rules only in the prompt, the 7B model ran 7 unsafe actions across 3 seeds. A read-only SQL authorizer plus an observation sanitizer brought that to 0, and task success didn't drop.
> - Retries barely matter for the 7B model (+2 points, not significant). For the 1.5B model they decide everything (+22 points).
>
> The 12% it still fails comes from skipped lookups and answering "$0" instead of "no data". Traces for every episode are in the repo.
>
> Code, eval and trace viewer: [repo link]
