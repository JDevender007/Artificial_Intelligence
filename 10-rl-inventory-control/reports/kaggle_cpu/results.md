# Results: final held-out evaluation

## Inventory-v0

Final score = mean over 50 held-out episodes per trained policy, then mean ± std over training seeds (t 95% CI). Heuristics are deterministic and tuned once on separate seeds.

| method | seeds | final mean ± std | 95% CI | IQM | AUC | fill rate | train s |
|---|---|---|---|---|---|---|---|
| random | 1 | -47.7 | n/a | -43.3 | n/a | 0.992 | n/a |
| base_stock {'S': 50} | 1 | 1,768.0 | n/a | 1,772.1 | n/a | 0.990 | n/a |
| dow_sS {'q': 0.8, 'batch': 35} | 1 | 2,331.1 | n/a | 2,331.8 | n/a | 0.989 | n/a |
| sS {'s': 38, 'S': 70} | 1 | 2,262.0 | n/a | 2,273.1 | n/a | 0.979 | n/a |
| dqn | 5 | 2,209.1 ± 102.6 | [2,081.6, 2,336.5] | 2,223.3 | 2,060.7 | 0.990 | 414 |
| dqn_nodouble | 5 | 2,244.4 ± 71.0 | [2,156.2, 2,332.6] | 2,247.3 | 2,074.6 | 0.990 | 414 |
| dqn_notarget | 5 | -2,346.1 ± 814.7 | [-3,357.7, -1,334.5] | -2,031.0 | -3,097.4 | 0.994 | 459 |
| sb3_dqn | 5 | 2,229.6 ± 47.3 | [2,170.9, 2,288.3] | 2,237.1 | 2,064.6 | 0.991 | 533 |
| sb3_ppo | 5 | 2,326.2 ± 5.2 | [2,319.6, 2,332.7] | 2,333.7 | 1,823.8 | 0.989 | 83 |

## CartPole-v1

Final score = mean over 50 held-out episodes per trained policy, then mean ± std over training seeds (t 95% CI). Heuristics are deterministic and tuned once on separate seeds.

| method | seeds | final mean ± std | 95% CI | IQM | AUC | solved seeds | median steps to solve | train s |
|---|---|---|---|---|---|---|---|---|
| dqn | 5 | 163.3 ± 93.2 | [47.6, 279.0] | 138.2 | 158.2 | 1/5 | 35,000 | 133 |
| dqn_nodouble | 5 | 276.3 ± 204.6 | [22.3, 530.3] | 248.0 | 163.1 | 3/5 | 47,500 | 135 |
| dqn_notarget | 5 | 9.4 ± 0.1 | [9.2, 9.5] | 9.4 | 9.9 | 0/5 | n/a | 140 |
| sb3_dqn | 5 | 342.7 ± 224.1 | [64.5, 620.9] | 408.6 | 225.6 | 5/5 | 35,000 | 130 |
| sb3_ppo | 5 | 500.0 ± 0.0 | [500.0, 500.0] | 500.0 | 374.7 | 5/5 | 15,000 | 62 |
