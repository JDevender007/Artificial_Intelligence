# Error analysis (seed 0, k=20)

## Hit rate of test positives by item popularity

Share of (user, liked test item) pairs found in the user's top-k, split by whether the item is a head item (top 10% by train positives), a tail item, or cold (no train ratings).

| model | head | tail (seen in train) | cold item |
|---|---|---|---|
| popularity | 0.161 (25607) | 0.000 (28795) | 0.000 (50) |
| bpr_mf | 0.157 (25607) | 0.005 (28795) | 0.000 (50) |
| two_tower | 0.114 (25607) | 0.015 (28795) | 0.000 (50) |
| two_tower_nofeat | 0.114 (25607) | 0.008 (28795) | 0.000 (50) |
| two_tower_nologq | 0.035 (25607) | 0.029 (28795) | 0.000 (50) |
| two_tower_inbatch | 0.116 (25607) | 0.016 (28795) | 0.000 (50) |

(counts in brackets are the number of pairs)

## Example users

Removed from the committed copy: it lists per-user rating histories with movie titles, which the MovieLens licence does not allow redistributing. Rerun the experiment to regenerate it locally.
