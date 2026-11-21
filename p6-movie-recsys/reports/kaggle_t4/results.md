# Results: ml1m (test period, 1188 users, mean ± std over seeds)

| model | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | coverage@20 | tail_share@20 | cold_item_recall@20 | seeds |
|---|---|---|---|---|---|---|---|---|---|
| popularity | 0.057 ± 0.000 | 0.092 ± 0.000 | 0.181 ± 0.000 | 0.217 ± 0.000 | 0.204 ± 0.000 | 0.058 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 3 |
| bpr_mf | 0.057 ± 0.002 | 0.097 ± 0.002 | 0.191 ± 0.002 | 0.209 ± 0.013 | 0.200 ± 0.010 | 0.257 ± 0.110 | 0.123 ± 0.096 | 0.000 ± 0.000 | 3 |
| two_tower | 0.053 ± 0.004 | 0.088 ± 0.005 | 0.166 ± 0.009 | 0.173 ± 0.006 | 0.168 ± 0.006 | 0.339 ± 0.013 | 0.242 ± 0.017 | 0.000 ± 0.000 | 3 |
| two_tower_nofeat | 0.044 ± 0.002 | 0.076 ± 0.002 | 0.153 ± 0.002 | 0.157 ± 0.002 | 0.151 ± 0.002 | 0.372 ± 0.076 | 0.271 ± 0.065 | 0.000 ± 0.000 | 3 |
| two_tower_nologq | 0.025 ± 0.001 | 0.045 ± 0.002 | 0.091 ± 0.007 | 0.082 ± 0.004 | 0.082 ± 0.004 | 0.679 ± 0.016 | 0.764 ± 0.017 | 0.000 ± 0.000 | 3 |
| two_tower_inbatch | 0.051 ± 0.003 | 0.087 ± 0.003 | 0.168 ± 0.002 | 0.176 ± 0.006 | 0.171 ± 0.006 | 0.350 ± 0.086 | 0.245 ± 0.084 | 0.007 ± 0.012 | 3 |

Training time (s): popularity 0, bpr_mf 9, two_tower 23, two_tower_nofeat 26, two_tower_nologq 34, two_tower_inbatch 18

Head = top 10% items by train positives (388 items). Cold items = no train ratings (221). popularity is deterministic: seeds identical.
