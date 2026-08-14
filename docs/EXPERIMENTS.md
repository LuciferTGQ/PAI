# Formal Experiments

## IB Medium M-1000

The formal run uses the released 1,000-trajectory training split and the
100-trajectory validation split. All trajectories contain 1,000 transitions.
The source and data checks recorded in `SOURCE_AUDIT.md` and `DATA_AUDIT.md`
were reused. No source document was reparsed or rendered, and training used the
already verified local dataset cache.

### World model setup

The Temporal Transformer consumes 30 chronological six-variable frames plus
the current three-dimensional action and predicts the next six-variable frame.
It has two encoder layers, `d_model=64`, four attention heads, a 128-unit feed-
forward block, and dropout 0.1. AdamW uses learning rate 3e-4 and weight decay
1e-5. Training uses batch size 512, AMP, gradient clipping at 1.0, seed 42, and
five epochs on the RTX 4060 Laptop GPU.

Five epochs equal about 9,770 gradient updates. The normalized validation MSE
by epoch was 0.033629, 0.032393, 0.033068, 0.032030, and 0.033447, so checkpoint
selection retained epoch 4. This is the best point under the declared budget,
not a claim of asymptotic convergence.

### Prediction results

Fresh-process evaluation used 100,000 one-step validation samples. Multi-step
evaluation used 9,600 starts, the true future action sequence, no true-state
reinjection, and no episode crossing.

| Metric | Result |
|---|---:|
| One-step MSE | 2.2927 |
| One-step MAE | 0.6285 |
| One-step RMSE | 1.5142 |
| One-step NRMSE | 0.1790 |

| Free-running horizon | MSE | NRMSE |
|---:|---:|---:|
| 1 | 2.4058 | 0.1881 |
| 5 | 5.2852 | 0.2233 |
| 10 | 11.3308 | 0.2964 |
| 20 | 13.7839 | 0.3335 |
| 50 | 21.5746 | 0.3936 |

Input ablations show that the result is not explained by copying the latest
frame: persistence MSE is 7.1968, shuffled-action MSE 9.0959, zero-action MSE
13.5649, and replacing the history by 30 copies of the latest frame yields MSE
3.0223. The full model reduces MSE 68.14% relative to persistence and uses both
the action and temporal history.

The selected 50-step visual rollout still smooths stochastic fatigue and can
drift substantially on consumption. This is an observed long-horizon limit;
the controller therefore uses a shorter five-step planning horizon.

### Official BC baseline

The fixed OfflineRL BC path uses two 256-unit LeakyReLU layers, a Gaussian NLL
objective, batch size 256, Adam 1e-3, and 100 epochs of 1,000 sampled updates.
The official summed batch-MSE validation criterion retained epoch 22; the
sample-weighted action MSE is 0.100154. Later epochs diverged, consistent with
the unnormalized observations and official 1e-3 learning rate. The best actor
was preserved exactly as the official code specifies.

### Frozen-world-model strategy

CEM uses horizon 5, population 64, 8 elites, two iterations, and a BC mean-
action warm start. It searches only inside `[-1, 1]^3`. The world model is
loaded in evaluation mode with every parameter frozen and is never updated.

Official IBGym online evaluation uses setpoint 70, classic reward, 1,000 steps,
and matched seeds 42 through 51.

| Policy | Evaluation | Episodes | Mean return | Std |
|---|---|---:|---:|---:|
| Released behavior trajectories | Offline reference | 1,000 | -282,884.61 | 1,197.29 |
| Official BC | Online matched seeds | 10 | -288,218.98 | 269.90 |
| Frozen-WM CEM | Online matched seeds | 10 | -278,270.08 | 5,326.65 |

WM-CEM improves over BC on all ten matched seeds. The paired mean improvement
is 9,948.90 return, with a 95% t interval of [6,033.37, 13,864.43] and paired
t-test p=0.000277. The large WM-CEM standard deviation and unusually strong
seed-49 result should remain visible rather than being hidden by the mean.

The BC mean action is unbounded by official design; 2.10% of its scalar actions
were clipped at the simulator boundary. CEM required no boundary clipping.
WM-CEM also exceeds the offline behavior-reference mean by 4,614.53, but that
comparison is descriptive only because the behavior-policy checkpoint is not
published and the released trajectories are not matched to the online seeds.

### Reproducibility artifacts

- `configs/world_model_ib_m1000.yaml`
- `configs/official_bc_ib_m1000.yaml`
- `configs/strategy_ib_m1000.yaml`
- `outputs/metrics/world_model_ib_m1000_*.json`
- `outputs/metrics/official_bc_ib_m1000_training.csv`
- `outputs/metrics/strategy_ib_m1000.json`
- `outputs/figures/*_ib_m1000_*.png`

Local ignored checkpoint SHA-256 values for this run are:

- world model: `AF3D196E5EB9A86F9ED42A7224FD8CA9D853BCC6C747BD2D53BA3B07020E0C1B`
- official BC: `A8C1A9E67AF26F47778F8E41A8F094DF19C11F1A78E7C50F21A829BE528F2D8D`
