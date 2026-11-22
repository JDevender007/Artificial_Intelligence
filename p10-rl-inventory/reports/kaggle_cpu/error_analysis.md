# Error analysis (seed-0 DQN vs tuned weekday (s,S), 50 held-out episodes; per-episode means)

| policy | profit | orders | units_ordered | fixed_cost | lost | wasted | holding | end_stock |
|---|---|---|---|---|---|---|---|---|
| dqn_seed0 | 2,204.7 | 35.1 | 1,000.9 | 350.8 | 8.9 | 2.0 | 241.1 | 51.0 |
| dow_sS | 2,331.1 | 25.5 | 993.2 | 254.8 | 10.5 | 2.4 | 213.8 | 44.4 |

| algo | mean final eval | mean peak eval | mean (peak - final) | seeds where final < 95% of peak |
|---|---|---|---|---|
| dqn | 2,237.9 | 2,327.6 | 89.7 | 1/5 |
| dqn_nodouble | 2,274.8 | 2,351.8 | 77.0 | 2/5 |
| sb3_dqn | 2,214.6 | 2,342.3 | 127.7 | 2/5 |
| sb3_ppo | 2,345.9 | 2,364.1 | 18.1 | 0/5 |
