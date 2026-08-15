# World Model and End-to-End Data Scaling

## Protocol

The architecture table contains MLP, GRU, LSTM, and the selected Transformer depth (Transformer-4L) at 100, 1,000, and 10,000 training trajectories. Every model keeps the common interface `history [B,30,6] + action [B,3] -> next_frame [B,6]` and uses the same normalization, trainer, loss, and validation-only checkpoint-selection rule:

1. find the best one-step NRMSE;
2. retain checkpoints within `1.10 x best_one_step_NRMSE`;
3. among retained checkpoints, minimize `mean(NRMSE_H5, NRMSE_H10, NRMSE_H20)`;
4. use H50 only as a stability diagnostic.

M-100 uses 50 epochs with checkpoints every 5 epochs. M-10000 uses 10 epochs with a checkpoint every epoch. No architecture at either scale satisfies the predeclared extension condition, so no run is extended to 75/100 or 20 epochs.

Checkpoint selection uses the official matched validation split for each data scale. Cross-scale reporting then evaluates the 12 selected checkpoints on the same largest validation file and uses one fixed target standard deviation computed from that common validation set. This avoids changing the NRMSE denominator across scales. The earlier per-model-normalization common metrics are preserved as a diagnostic and are not used in the formal scaling figure.

## Why low epoch counts still contain many updates

Epoch counts are not comparable across data scales. With batch size 512:

| Training scale | Samples per epoch | Updates per epoch (approx.) | Scheduled epochs | Total sample presentations | Total updates (approx.) |
|---:|---:|---:|---:|---:|---:|
| M-100 | 100,000 | 196 | 50 | 5 million | 9,800 |
| M-1000 | 1,000,000 | 1,954 | 50 | 50 million | 97,700 |
| M-1000 legacy 100e | 1,000,000 | 1,954 | 100 | 100 million | 195,400 |
| M-10000 | 10,000,000 | 19,532 | 10 | 100 million | 195,320 |

Thus, ten M-10000 epochs expose the optimizer to approximately the same number of samples and minibatch updates as 100 M-1000 epochs. Fast epoch-level convergence is not evidence that only a few gradient steps occurred. The models are also small (114k-227k parameters), the task is normalized supervised regression, and training uses mixed precision on GPU.

## Selected checkpoints on matched validation

Values are `selected epoch / one-step NRMSE / mean H5-H10-H20 NRMSE`.

| Architecture | M-100 | M-1000 | M-10000 |
|---|---:|---:|---:|
| MLP | e35 / 0.1964 / 0.2772 | e2 / 0.1869 / 0.2668 | e1 / 0.2013 / 0.3966 |
| GRU | e15 / 0.1936 / 0.2268 | e7 / 0.1788 / 0.2386 | e1 / 0.1872 / 0.3257 |
| LSTM | e35 / 0.1909 / 0.2478 | e13 / 0.1790 / 0.2735 | e4 / 0.2057 / 0.3231 |
| Transformer-4L | e19 / 0.1892 / 0.3097 | e40 / 0.1906 / 0.2432 | e6 / 0.1918 / 0.2570 |

This table demonstrates why a validation-loss minimum or final epoch is insufficient. For example, M-100 GRU reaches its lowest validation MSE at epoch 48, but the deployment protocol selects epoch 15. At M-10000, LSTM and Transformer-4L have their lowest one-step validation MSE at epoch 1, but rollout-aware selection chooses epochs 4 and 6 respectively.

## Formal 4 x 3 comparison on one common validation

Values are `one-step NRMSE / mean H5-H10-H20 NRMSE`, using the same validation trajectories and fixed NRMSE denominator for all cells.

| Architecture | M-100 | M-1000 | M-10000 |
|---|---:|---:|---:|
| MLP | 0.1900 / 0.3757 | 0.1899 / 0.3852 | 0.2191 / 0.5824 |
| GRU | 0.1851 / 0.2923 | **0.1742** / 0.3318 | 0.1884 / 0.4613 |
| LSTM | 0.1840 / 0.3126 | 0.1761 / 0.4027 | 0.2214 / 0.4439 |
| Transformer-4L | **0.1819** / 0.4253 | 0.1831 / 0.3378 | **0.1879 / 0.2951** |

More data is not uniformly beneficial under one fixed capacity and optimization schedule. Transformer-4L shows the clearest rollout scaling, improving from 0.4253 to 0.3378 to 0.2951. MLP and the recurrent models do not improve monotonically at M-10000. Plausible mechanisms are capacity/optimization mismatch under a broader and more heterogeneous training distribution, rather than insufficient raw gradient-step count. This is an empirical finding, not a reason to retroactively change the M-1000 deployment model.

## Fixed-architecture, fixed-strategy end-to-end scaling

The end-to-end comparison follows the latest protocol: the World Model architecture is fixed to the M-1000-selected GRU and the Strategy is fixed to CEM H=10. The official M-1000 BC checkpoint is also fixed as the CEM proposal prior so that only the World Model training scale changes. Checkpoints are selected on validation only. Simulator seeds 47-51 are used for reporting, not tuning.

| World Model training scale | Selected GRU epoch | Held-out return mean | Std. dev. | Median |
|---:|---:|---:|---:|---:|
| M-100 | 15 | -279,307.6 | 13,390.3 | -272,257.6 |
| M-1000 | 7 | **-268,168.8** | 750.0 | -268,308.2 |
| M-10000 | 1 | -268,291.3 | **464.0** | -268,290.1 |

M-1000 and M-10000 are effectively tied on these five matched seeds: the paired mean difference (M-10000 minus M-1000) is -122.5 return units with a 95% paired t interval of approximately [-1,214.9, 969.9]. With only five seeds, this interval is descriptive rather than a strong inferential claim. M-100 is less reliable and contains one severe seed-51 failure (-306,007.2).

The common-validation GRU rollout NRMSE does not map monotonically to CEM reward: M-10000 has worse aggregate GRU rollout NRMSE than M-1000 but nearly identical simulator return. This reinforces two boundaries:

- simulator reward must not select or modify the World Model;
- short-horizon planner performance depends on which prediction errors matter along planned trajectories, not only a global aggregate validation error.

## Main conclusion

For M-1000 deployment, the selected Frozen World Model remains GRU epoch 7 and the selected Strategy remains CEM H=10. M-10000 does not justify changing that decision: it is much more expensive, its fixed-GRU control result is statistically indistinguishable from M-1000 on the current held-out seeds, and its GRU dynamics metrics are worse on common validation. Separately, Transformer-4L is the architecture that benefits most consistently from additional World Model data and should be highlighted in the architecture-scaling analysis.

## Reproducible artifacts

- Combined model table: `outputs/metrics/world_model_data_scaling_models.csv`
- Fixed-denominator common validation: `outputs/metrics/world_model_data_scaling_common_validation_fixed_std.csv`
- Full checkpoint metrics: `outputs/metrics/world_model_data_scaling_checkpoints.csv`
- End-to-end metrics: `outputs/metrics/strategy_end_to_end_data_scaling.json`
- Editable figures: `outputs/figures/world_model_data_scaling.svg` and `outputs/figures/strategy_end_to_end_data_scaling.svg`
