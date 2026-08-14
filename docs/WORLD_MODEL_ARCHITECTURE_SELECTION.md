# M-1000 World Model Architecture and Checkpoint Selection

## Result

`GRU, epoch 7` is the selected M-1000 deployment dynamics model. It is now the
single Frozen World Model used by downstream CEM, iCEM, MPPI, and MB-PPO work.
The choice uses validation dynamics only; no strategy return or simulator
episode reward entered either checkpoint or architecture selection.

The closest alternative is Transformer-4L at epoch 40. Its short-rollout mean
NRMSE is 0.243177 versus 0.238566 for GRU, so GRU is 1.90% lower. Transformer-4L
has the lower H50 diagnostic error, but H50 was declared diagnostic-only before
evaluation and therefore cannot override the H5/H10/H20 deployment criterion.

## Locked protocol

All models implement the same interface:

```text
history [B, 30, 6] + action [B, 3] -> next_frame [B, 6]
```

The four new architectures use the same M-1000 training data, M-100 validation
data, training-data normalization, AdamW optimizer, learning rate 3e-4, weight
decay 1e-5, batch size 512, seed 42, mixed precision, and trainer. Each was run
for 50 epochs with a checkpoint every 5 epochs. Transformer-4L changes only the
encoder depth from 2 to 4 while retaining d_model=64, 4 heads, and FFN=128.

Checkpoint selection is performed within each architecture, then the identical
rule is applied across the five architecture-level candidates:

1. Find the lowest one-step NRMSE.
2. Keep candidates with `one_step_NRMSE <= 1.10 * best_one_step_NRMSE`.
3. Select the candidate with the lowest mean of NRMSE at H5, H10, and H20.
4. Report H50 only as long-rollout stability diagnosis.

The global best one-step NRMSE is 0.178800, giving a threshold of 0.196680.
Every architecture-level candidate passes the threshold.

## Model table

| Architecture | Parameters | Selected epoch | One-step NRMSE | H5 | H10 | H20 | Mean H5/H10/H20 | H50 diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **GRU** | 177,030 | **7** | **0.178800** | **0.199426** | **0.225387** | 0.290886 | **0.238566** | 0.414652 |
| Transformer-4L | 158,086 | 40 | 0.190614 | 0.205452 | 0.234798 | **0.289281** | 0.243177 | 0.365822 |
| Transformer-2L | 91,142 | 100 | 0.196003 | 0.217333 | 0.252403 | 0.296969 | 0.255568 | **0.318226** |
| MLP | 114,438 | 2 | 0.186938 | 0.220988 | 0.257235 | 0.322114 | 0.266779 | 0.494452 |
| LSTM | 227,462 | 13 | 0.179008 | 0.214211 | 0.263457 | 0.342903 | 0.273524 | 0.546512 |

Transformer-2L reuses the existing epoch 4, 11, and 100 results and checkpoints.
Under the same within-model rule, epoch 100 remains eligible because its
one-step NRMSE (0.196003) is below 1.10 times Transformer-2L's best one-step
value (0.178498), and it has the lowest H5/H10/H20 mean of those three points.

## Epoch decision

No new architecture meets the predeclared condition for extension to epoch 75:
the selected checkpoint must be near epoch 50 and H5/H10/H20 must still be
clearly improving. MLP, GRU, and LSTM select epochs 2, 7, and 13. Transformer-4L
selects epoch 40, while epochs 45 and 50 worsen the short-rollout mean from
0.243177 to 0.263635 and 0.281992. Extending these runs would add compute without
evidence that the selection frontier is still moving at the end of training.

## BC source check

| Source | BC learning rate | Location and behavior |
|---|---:|---|
| `[PAPER]` NeoRL v2 | 3e-4 | Appendix BC hyperparameters; batch 256, 100K steps, lowest validation NLL |
| `[OFFICIAL CODE]` OfflineRL commit `807933a87f77529f17bd81ac64d717aad89f5cdf` | 1e-3 | `offlinerl/config/algo/bc_config.py`, `actor_lr = 1e-3`; consumed by Adam in `offlinerl/algo/modelfree/bc.py` |
| `[VERIFIED LOCALLY]` current project | 1e-3 | `configs/official_bc_ib_m1000.yaml`, faithfully reproducing the fixed official code configuration |

The paper/code discrepancy remains documented. Because the current implementation
matches the fixed official implementation, BC is retained without retraining.

## Artifacts and review limits

- Full checkpoint-level source data: `outputs/metrics/world_model_ib_m1000_architecture_checkpoints.csv`
- Architecture-level table: `outputs/metrics/world_model_ib_m1000_architecture_models.csv`
- Selection manifest: `outputs/metrics/world_model_ib_m1000_architecture_selection.json`
- Editable figure: `outputs/figures/world_model_ib_m1000_architecture_selection.svg`
- PDF and PNG exports use the same Python rendering path.

The architecture study uses one training seed. Selection is evaluated on 100
held-out validation trajectories and 9,600 common rollout starts; uncertainty
across independently trained World Model seeds is not estimated here. Downstream
strategy evaluation will use held-out simulator seeds, but those results will not
be allowed to revise this Frozen World Model choice.
