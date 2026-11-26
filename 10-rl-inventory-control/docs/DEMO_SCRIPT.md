# P10 60-second demo script

1. **(0-10 s)** "This is a shop simulator: a 2-day lead time, a fixed cost per order, lost sales, and a weekend demand peak. I trained a DQN I wrote from scratch, plus Stable-Baselines3 DQN and PPO, and compared them against classical inventory policies."
2. **(10-25 s)** Run `python scripts/play.py --run-dir outputs/main --days 14 --seed 100000`. Scroll through both 14-day logs. "Same demand, same starting stock. The weekday (s, S) policy made 278.8 over these 14 days and the DQN made 186.3. Look at the order column: the DQN orders more often, in smaller lots."
3. **(25-40 s)** Show the Inventory results table. "Five seeds, 50 held-out episodes. PPO ties the informed heuristic, 2,326 vs 2,331, with a bootstrap CI of −16 to +7. PPO beats the textbook (s, S) by 64. My DQN is 441 above base-stock but 122 below the heuristic."
4. **(40-52 s)** Open `reports/kaggle_cpu/error_analysis.md`. "The gap is batching. Both policies buy about 1,000 units, but the DQN places 35 orders instead of 25, so it pays roughly 96 extra in fixed fees. Its lost sales are actually lower."
5. **(52-60 s)** Open `curves_Inventory-v0.png`. "Remove the target network and DQN diverges to −2,346. Double-DQN made no measurable difference. PPO is the stable one: std 5 across seeds against 103 for DQN."
