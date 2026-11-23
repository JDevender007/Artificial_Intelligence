# Hierarchical paired bootstrap

5000 resamples of training seeds (per side) and shared evaluation episodes.

## Inventory-v0

| a - b | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| dqn - dqn_nodouble | -35.34 | [-139.34, +54.58] | 0.747 |
| dqn - dqn_notarget | +4,555.15 | [+4,131.97, +5,306.81] | 0.000 |
| dqn_nodouble - dqn_notarget | +4,590.49 | [+4,183.33, +5,331.23] | 0.000 |
| dqn - sb3_dqn | -20.54 | [-114.82, +57.60] | 0.649 |
| sb3_ppo - sb3_dqn | +96.56 | [+58.62, +134.17] | 0.000 |
| sb3_ppo - dqn | +117.11 | [+51.86, +208.16] | 0.000 |
| dqn - base_stock | +441.01 | [+349.61, +505.29] | 0.000 |
| dqn - sS | -52.91 | [-146.80, +13.38] | 0.908 |
| dqn - dow_sS | -122.04 | [-213.88, -56.99] | 1.000 |
| dqn_nodouble - base_stock | +476.36 | [+419.82, +529.58] | 0.000 |
| dqn_nodouble - sS | -17.57 | [-75.18, +37.04] | 0.734 |
| dqn_nodouble - dow_sS | -86.69 | [-144.12, -33.88] | 1.000 |
| dqn_notarget - base_stock | -4,114.14 | [-4,846.00, -3,736.14] | 1.000 |
| dqn_notarget - sS | -4,608.06 | [-5,342.67, -4,224.67] | 1.000 |
| dqn_notarget - dow_sS | -4,677.19 | [-5,410.80, -4,297.05] | 1.000 |
| sb3_dqn - base_stock | +461.56 | [+425.05, +499.83] | 0.000 |
| sb3_dqn - sS | -32.37 | [-70.93, +8.97] | 0.935 |
| sb3_dqn - dow_sS | -101.49 | [-138.50, -61.95] | 1.000 |
| sb3_ppo - base_stock | +558.12 | [+547.46, +568.62] | 0.000 |
| sb3_ppo - sS | +64.20 | [+47.76, +80.19] | 0.000 |
| sb3_ppo - dow_sS | -4.93 | [-16.37, +7.02] | 0.803 |

## CartPole-v1

| a - b | diff | 95% CI | P(diff <= 0) |
|---|---|---|---|
| dqn - dqn_nodouble | -113.02 | [-294.22, +51.53] | 0.897 |
| dqn - dqn_notarget | +153.94 | [+85.49, +228.89] | 0.000 |
| dqn_nodouble - dqn_notarget | +266.96 | [+114.57, +419.32] | 0.000 |
| dqn - sb3_dqn | -179.37 | [-356.73, +9.79] | 0.968 |
| sb3_ppo - sb3_dqn | +157.32 | [+0.00, +348.98] | 0.082 |
| sb3_ppo - dqn | +336.69 | [+257.42, +404.47] | 0.000 |
