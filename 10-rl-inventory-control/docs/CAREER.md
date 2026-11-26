# Resume, LinkedIn, repo name

**GitHub location:** `JDevender007/Artificial_Intelligence/10-rl-inventory-control` (monorepo). Standalone name if split out: `rl-inventory-control`

**Resume bullet**
- Built a custom Gymnasium inventory environment (lead time, fixed order cost, lost sales, weekly seasonality) and a from-scratch PyTorch DQN with ablations. Benchmarked them against SB3 DQN/PPO and grid-tuned (s, S) policies over 5 seeds × 50 held-out episodes, with hierarchical bootstrap CIs. PPO matched the informed heuristic (2,326 vs 2,331, CI −16 to +7) and beat the textbook (s, S) by +64. Removing the target network collapsed DQN from +2,209 to −2,346 profit per episode.

**LinkedIn post (draft)**
> I trained RL agents to run a simulated shop's inventory. The best one tied a 1950s-style reorder rule.
>
> The setup: a custom Gymnasium environment with a 2-day lead time, a fixed cost per order, lost sales and a weekend demand peak. On the RL side, a DQN I wrote from scratch, plus Stable-Baselines3 DQN and PPO. On the other side, heuristics tuned on separate seeds. 5 training seeds, 50 held-out episodes, bootstrap CIs.
> - PPO: 2,326 profit per episode. The weekday-aware (s, S) rule: 2,331. Statistically a tie.
> - PPO *does* beat the textbook (s, S) rule by +64, because it learned the weekly cycle from reward alone.
> - My DQN is 441 above base-stock, but 122 below the best rule. The error analysis shows why: it orders 35 times a quarter instead of 25, and the fixed fees add up.
> - Delete the target network and DQN diverges to −2,346. Double-DQN made no measurable difference.
>
> The lesson: always build the strong boring baseline first. Without it, "RL beats base-stock by 441" looks like a win.
>
> Code, configs and every seed's curve: [repo link]
