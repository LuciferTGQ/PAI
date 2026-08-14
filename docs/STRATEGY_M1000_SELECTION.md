# M-1000 Strategy Selection

## Locked inputs

- Frozen World Model: `outputs/checkpoints/world_model_ib_m1000_gru_best.pt` (GRU, epoch 7), selected only by validation dynamics metrics.
- Planning horizon: 10, selected on development seeds 42-46 by the predeclared one-standard-error rule over CEM horizons 5/10/20.
- Development seeds: 42-46. Held-out reporting seeds: 47-51.
- The World Model checkpoint and normalization remained immutable for CEM, iCEM, MPPI, and both MB-PPO variants.

## Development results

| Strategy | Mean return | Std. dev. | Interpretation |
|---|---:|---:|---|
| CEM H=10 | -269,373.9 | 486.9 | Selected main Strategy |
| MB-PPO w/ behavior KL | -284,714.8 | 173.6 | Stable, improves over BC, below CEM |
| Official BC | -288,175.2 | 201.9 | Reused reference |
| iCEM H=10 | -337,220.2 | 20,823.7 | Below CEM under fixed low-cost settings |
| MB-PPO w/o behavior KL | -883,708.6 | 499.7 | Severe model exploitation |
| MPPI H=10 | -1,002,582.7 | 159,821.6 | Unstable model exploitation / destructive updates |

The MPPI effective-sample-size correction increased numerical sample diversity but reduced development performance further (mean -2,600,555.7). It is retained as a diagnostic and was not advanced to held-out reporting.

## MB-PPO implementation audit

NeoRL v2 specifies a frozen transition model, dataset-sampled starting states, 10-step model rollouts, BC policy initialization, two-layer 256-unit policy/value MLPs, tanh actions, Adam at `3e-4`, 5K gradient steps, and an optional `D_KL(pi_behavior || pi)` penalty. The fixed OfflineRL commit has no MB-PPO implementation, and the paper does not disclose the penalty coefficient. The project therefore uses standard clipped PPO with an explicit coefficient of `1.0`; the no-KL variant sets only that coefficient to zero.

At step 5,000, the with-KL minibatch behavior KL was approximately `0.068`, while the no-KL variant reached approximately `1.96e9` after much larger intermediate excursions. The corresponding development simulator results separate cleanly: -284.7k with KL versus -883.7k without KL.

## One-time held-out report

Strategy selection was locked as CEM H=10 before opening seeds 47-51.

| Strategy | Held-out mean | Std. dev. | Held-out rank |
|---|---:|---:|---:|
| CEM H=10 | -268,168.8 | 750.0 | 1 |
| MB-PPO w/ behavior KL | -284,831.6 | 241.2 | 2 |
| Official BC | -288,262.8 | 317.9 | reference |
| iCEM H=10 | -339,615.6 | 19,691.6 | 3 |
| MB-PPO w/o behavior KL | -884,692.0 | 796.8 | 4 |
| MPPI H=10 | -931,595.1 | 250,166.7 | 5 |

The strategy ranking is unchanged between development and held-out seeds. CEM H=10 is the fixed M-1000 Strategy for subsequent end-to-end data scaling. This decision does not alter the validation-selected GRU World Model.

## Reproducible artifacts

- MB-PPO config: `configs/mbppo_ib_m1000_gru.yaml`
- Held-out config: `configs/strategy_heldout_ib_m1000_gru.yaml`
- Development metrics: `outputs/metrics/strategy_ib_m1000_gru_mbppo.json`
- Held-out metrics: `outputs/metrics/strategy_ib_m1000_gru_heldout.json`
- Editable figures: `outputs/figures/strategy_ib_m1000_gru_mbppo.svg` and `outputs/figures/strategy_ib_m1000_gru_heldout.svg`
