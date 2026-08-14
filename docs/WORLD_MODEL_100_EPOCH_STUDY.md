# World Model 100-Epoch Study

## Question

The original formal M-1000 run trained for five epochs and retained epoch 4.
This study holds the data, seed, model, batch size, optimizer, learning rate,
normalization, AMP, and evaluation splits fixed, changes only the maximum
training length to 100 epochs, disables early termination for the study, and
saves both the one-step-validation best checkpoint and the epoch-100 checkpoint.

The run completed all 100 epochs in 3,235 seconds. A vectorized Dataset batch
fetch was introduced for throughput; an exact-equality test confirms that it
returns the same tensors as individual sample fetching.

## Training trend

| Checkpoint | Train normalized MSE | Validation normalized MSE |
|---|---:|---:|
| Epoch 4, original best | 0.031602 | 0.032030 |
| Epoch 5 | 0.031146 | 0.033447 |
| Epoch 11, 100-run one-step best | 0.029811 | **0.031867** |
| Epoch 100, last | 0.021061 | 0.038407 |

Epoch 11 improves validation MSE by 0.51% relative to epoch 4. After epoch 11,
training MSE keeps falling while validation MSE trends upward; the final
generalization gap is 0.017345 and the validation slope over the last 20 epochs
is positive. Under the one-step normalized objective, epoch 100 is overfit.

This also shows that patience 5 can stop too early: after the epoch-4 best, the
next new low appears at epoch 11. Patience around 10–15 is safer for one-step
checkpoint selection.

## Prediction behavior

| Checkpoint | One-step MSE | H5 MSE | H10 MSE | H20 MSE | H50 MSE |
|---|---:|---:|---:|---:|---:|
| Epoch 4 | 2.2927 | 5.2852 | 11.3308 | 13.7839 | 21.5746 |
| Epoch 11 | **2.2741** | 4.4820 | 8.5252 | 10.7349 | 31.0240 |
| Epoch 100 | 2.7614 | **4.3033** | **6.2892** | **6.4757** | **10.6282** |

The one-step best is not the best free-running model. Epoch 100 is worse on
one-step validation but markedly more stable after repeated autoregressive
updates. This explains why selecting only by one-step MSE is misaligned with a
receding-horizon controller.

## Matched-seed simulator results

The official BC policy, CEM settings, environment, ten seeds, and 1,000-step
episode horizon are identical across the three world-model checkpoints.

| Policy/model | Mean return | Std |
|---|---:|---:|
| Official BC | -288,218.98 | 269.90 |
| WM-CEM, epoch 4 | -278,270.08 | 5,326.65 |
| WM-CEM, epoch 11 | -270,134.09 | 7,192.70 |
| WM-CEM, epoch 100 | **-267,337.61** | **398.54** |

Epoch 11 improves over epoch 4 by 8,135.99 mean return (8/10 seed wins,
paired p=0.0101). Epoch 100 improves over epoch 4 by 10,932.47 (9/10 wins,
paired p=0.000176). Epoch 100 exceeds epoch 11 by 2,796.48 on average and is
far less variable, but the paired difference is not statistically significant
with ten episodes (6/10 wins, p=0.265; 95% CI includes zero).

## Decision

Five epochs are not enough for the downstream world-model control objective.
They are adequate for a quick smoke run, but they leave meaningful short-
horizon and simulator performance unrealized.

Running longer is useful, but blindly selecting the lowest one-step validation
loss is not. For the current five-step CEM controller, epoch 100 is the stronger
provisional planning model because it has the best H5–H50 rollout behavior and
the highest, most stable real-simulator return. Epoch 11 remains the correct
checkpoint when the stated objective is one-step prediction.

Do not extend beyond 100 epochs yet under the same fixed learning rate. The
one-step curve is already worsening, and this study saved no intermediate
rollout-selected checkpoints between 11 and 100, so it cannot show whether the
planning optimum occurs at 60, 80, 100, or later. The next controlled run should:

1. save a checkpoint every five epochs;
2. select on held-out H5/H10 rollout error, with fatigue/consumption or predicted
   reward reported separately;
3. reduce learning rate after the one-step plateau instead of continuing at a
   fixed 3e-4;
4. repeat training across several model seeds before treating epoch 100 as a
   universal optimum.

Until that run, use the epoch-100 checkpoint for the existing CEM pipeline and
retain epoch 11 for one-step prediction reporting.

## Artifacts

- Config: `configs/world_model_ib_m1000_100e.yaml`
- Recommended controller config: `configs/strategy_ib_m1000_100e.yaml`
- Full history: `outputs/metrics/world_model_ib_m1000_100e_training.csv`
- Consolidated metrics: `outputs/metrics/world_model_ib_m1000_100e_analysis.json`
- Consolidated figure: `outputs/figures/world_model_ib_m1000_100e_analysis.png`
- Best checkpoint SHA-256: `D3912AE1E0EA22D5C43DCE0B13085AC66C12B631F436438A9F45660EF9E05457`
- Epoch-100 checkpoint SHA-256: `846C72A4EB9B5018D0738126C77497511A58E897267AFD9A0ABAD8B89C7365B7`
