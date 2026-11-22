# Paired bootstrap over tasks (task score = success averaged over seeds)

2000 resamples. diff = agent_full minus the other agent.

| model | vs | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|---|
| qwen7b_nf4 | direct | +0.633 | [+0.456, +0.800] | 0.000 |
| qwen7b_nf4 | agent_noguard | +0.078 | [+0.022, +0.156] | 0.004 |
| qwen7b_nf4 | agent_noretry | +0.022 | [+0.000, +0.056] | 0.123 |
| qwen1.5b | direct | +0.189 | [+0.089, +0.300] | 0.000 |
| qwen1.5b | agent_noguard | +0.044 | [-0.078, +0.167] | 0.257 |
| qwen1.5b | agent_noretry | +0.222 | [+0.122, +0.322] | 0.000 |
