# P10 interview Q&A

**1. Why would anyone use RL here if a heuristic ties it?**
In this environment the heuristic is *informed*. It reads the true Poisson rates and the weekday from the simulator, and (s, S) structure is close to optimal for this kind of problem. RL has to learn the seasonality from reward. PPO matching it (−4.9, CI [−16.4, +7.0]) and beating the seasonality-blind (s, S) by +64 shows the agent learned that structure. RL earns its keep when the dynamics have no clean analytical policy: multiple SKUs sharing capacity, substitution, perishability, or non-stationary demand. Even then, I'd ship it next to the heuristic in shadow mode first.

**2. How did you make the comparison fair?**
- **Separate seed pools:** heuristic tuning uses 200000+, learning-curve evals 50000+, and the final score 50 episodes at 100000+, which nothing is tuned or selected on.
- **Same budget, last checkpoint:** every learner gets the same step budget and is scored at its last checkpoint.
- **Across seeds:** 5 training seeds with t-intervals. Differences use a hierarchical paired bootstrap: training seeds are resampled per side, and the evaluation episodes are shared, so episode difficulty cancels out.
- **The known bias:** the heuristics were grid-tuned and the RL hyperparameters weren't. That bias favours the baseline, and I report it.

**3. What did the ablations show?**
- **The target network is essential.** Without it, DQN goes from +2,209 to −2,346 on inventory and from 163 to 9.4 on CartPole. The no-target agent has the *highest* fill rate (0.994), because its bootstrapped Q-values blow up and it over-orders until waste and holding costs swamp revenue.
- **Double-DQN gave no measurable gain:** −35 [−139, +55] on inventory and −113 [−294, +52] on CartPole. Overestimation bias is probably small with 9 actions and Huber-clipped targets, and 5 seeds can't resolve a small effect.

**4. Why does your DQN lose to the heuristic?**
I replayed the seed-0 DQN and the heuristic on the 50 held-out episodes and broke profit into its components.
- **Same quantity, more orders:** they buy about the same total (1,001 vs 993 units), but the DQN uses 35.1 orders against 25.5, so it pays about 96 more in fixed fees and 27 more in holding. Its lost sales are actually lower.
- **Why that's hard for DQN:** "order less often, in bigger lots" means paying a fixed cost today for sales spread over the next several days. That is a credit-assignment problem, and an ε-greedy value learner handles it badly.
- **Unstable checkpoints:** value-based agents end on average 78–128 profit below their own peak eval (seed 0 peaks at 2,330 and dips to 1,845). PPO ends 18 below its peak.

**5. What would you do next?**
- Give every learner an equal-budget hyperparameter sweep on the tuning seeds, and select the best checkpoint on the curve seeds, not the held-out ones.
- Try action parametrisation that makes batching natural: actions as an order-up-to level rather than a quantity, or (s, S)-parametrised policies learned by policy gradient.
- Model demand censoring: in reality you see sales, not demand.
- For deployment: run shadow mode next to the current planner, cap order sizes, alert when the policy drifts from the heuristic by more than X%, and keep a human override on every order.
