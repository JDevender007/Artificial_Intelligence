# P6 interview questions

**1. Your two-tower model lost to BPR-MF and tied with popularity. Why ship it at all?**
Because it answers a question MF cannot: what to show a user with no training history. MF gives such a user the popularity list, while the two-tower embeds demographics and a few liked movies. I report the accuracy loss as it is: −0.009 recall@20 vs BPR-MF, 95% CI [−0.016, −0.003]. On this split, BPR-MF is the better model for known users. A production system would route known users to MF and cold users to the tower, or blend them.

**2. Why a global time split instead of leave-one-out?**
With leave-one-out or a random split, a model can train on ratings made after the test ratings. That leaks future popularity and inflates the scores. A global cutoff means every model sees only the past. The price is a small, hard test set: 1,188 users, 195 of them with no train ratings, and 50 positives on movies that did not exist in train.

**3. What does logQ correction do, and how do you know it matters?**
With in-batch negatives, an item is sampled as a negative roughly in proportion to its popularity, so the softmax learns to push popular items down. Subtracting log q(item) from each logit undoes that sampling bias. The ablation without it drops recall@20 from 0.088 to 0.045, a gap of +0.043 with CI [+0.037, +0.049]. It also moves 76% of recommendations into the tail. So logQ is effectively a popularity dial.

**4. Recall went down but coverage went up. Which is right?**
Both are measurements; which one matters is a product decision. The test period rewards popularity: 47% of liked test pairs are on the top-10% items. The two-tower finds 3× more tail hits than BPR-MF (0.015 vs 0.005) and covers 34% of the catalogue at k=20, against 6% for popularity. Offline recall is biased toward what users were already shown. An online A/B test, or at least a popularity-debiased offline metric, would be needed to choose.

**5. Why HNSW if the catalogue has only 3,883 items?**
At this size I would not need it: flat search is exact and takes 0.05 ms. I benchmarked HNSW to show the trade-off: ef=32 gives 0.990 recall vs exact at about half the latency, and ef=128 gives 0.9998. At 10⁶ items flat search grows linearly while HNSW stays roughly logarithmic, and that is where the ef setting becomes a real recall/latency knob.
