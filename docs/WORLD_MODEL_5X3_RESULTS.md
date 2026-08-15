# Fair 5 x 3 World Model Matrix

## Protocol

All candidate checkpoints are evaluated on `data/raw/ib-medium-1000-val.npz`
with one fixed target standard deviation shared across every dataset scale and
architecture. Selection uses validation dynamics only:

1. select a checkpoint within each architecture;
2. retain architecture candidates with one-step NRMSE no greater than 1.10
   times the best one-step NRMSE at that data scale;
3. select the lowest mean of H5, H10, and H20;
4. use H50 only as a stability diagnostic.

Simulator return and Strategy metrics are not used. The legacy Transformer-2L
checkpoints are excluded. The fair Transformer-2L differs from Transformer-4L
only in `num_layers: 2` versus `4`.

## Selected checkpoint for each architecture

| Scale | Architecture | Epoch | Params | One-step | H5 | H10 | H20 | Mean H5/H10/H20 | H50 | Total train seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M-100 | MLP | 35 | 114438 | 0.190034 | 0.282287 | 0.380479 | 0.464356 | 0.375707 | 0.825036 | 62.57 |
| M-100 | GRU | 35 | 177030 | 0.179550 | 0.222830 | 0.272722 | 0.374375 | 0.289976 | 0.649034 | 87.97 |
| M-100 | LSTM | 35 | 227462 | 0.184018 | 0.229438 | 0.302151 | 0.406336 | 0.312642 | 0.610704 | 89.98 |
| M-100 | Transformer-2L | 35 | 91142 | 0.178505 | 0.314228 | 0.456075 | 0.638382 | 0.469562 | 0.898737 | 170.06 |
| M-100 | Transformer-4L | 19 | 158086 | 0.181878 | 0.300127 | 0.434725 | 0.541004 | 0.425285 | 0.710422 | 201.25 |
| M-1000 | MLP | 2 | 114438 | 0.189903 | 0.274016 | 0.371128 | 0.510440 | 0.385195 | 0.865689 | 673.19 |
| M-1000 | GRU | 7 | 177030 | 0.174181 | 0.234083 | 0.306935 | 0.454294 | 0.331771 | 0.720818 | 852.63 |
| M-1000 | LSTM | 13 | 227462 | 0.176066 | 0.272034 | 0.381213 | 0.554738 | 0.402662 | 0.975172 | 851.67 |
| M-1000 | Transformer-2L | 20 | 91142 | 0.182634 | 0.243219 | 0.305429 | 0.404560 | 0.317736 | 0.774474 | 1549.57 |
| M-1000 | Transformer-4L | 40 | 158086 | 0.183119 | 0.244653 | 0.318646 | 0.450139 | 0.337813 | 0.658066 | 1959.93 |
| M-10000 | MLP | 1 | 114438 | 0.219098 | 0.412445 | 0.596594 | 0.738086 | 0.582375 | 0.856714 | 1515.86 |
| M-10000 | GRU | 1 | 177030 | 0.188431 | 0.292522 | 0.422667 | 0.668718 | 0.461303 | 1.201179 | 1978.03 |
| M-10000 | LSTM | 2 | 227462 | 0.214821 | 0.326690 | 0.423084 | 0.585446 | 0.445074 | 1.022326 | 1946.65 |
| M-10000 | Transformer-2L | 4 | 91142 | 0.205012 | 0.257820 | 0.297535 | 0.293473 | 0.282943 | 0.509893 | 3585.50 |
| M-10000 | Transformer-4L | 6 | 158086 | 0.187894 | 0.246499 | 0.290357 | 0.348394 | 0.295083 | 0.581294 | 4330.21 |

## Best World Model by data scale

| Dataset | BestWM | Epoch | Eligible architectures | One-step | Mean H5/H10/H20 | H50 |
|---|---|---:|---|---:|---:|---:|
| M-100 | GRU | 35 | all five | 0.179550 | 0.289976 | 0.649034 |
| M-1000 | Transformer-2L | 20 | all five | 0.182634 | 0.317736 | 0.774474 |
| M-10000 | Transformer-2L | 4 | GRU, Transformer-2L, Transformer-4L | 0.205012 | 0.282943 | 0.509893 |

The M-10000 Transformer-2L remains eligible because 0.205012 is below the
1.10 threshold of 0.206683, then wins on H5/H10/H20 mean. No model meets the
predefined extension condition.

## Transformer depth analysis

- M-100: 4L worsens one-step by 0.003372 but improves H5/H10/H20 mean by
  0.044276 and H50 by 0.188315 relative to 2L.
- M-1000: 4L worsens one-step by 0.000485 and H5/H10/H20 mean by 0.020076,
  while improving H50 by 0.116408.
- M-10000: 4L improves one-step by 0.017118 and improves H5/H10, but worsens
  H20 by 0.054921, H50 by 0.071401, and the selection-horizon mean by 0.012141.

Increasing Transformer depth does not stably improve both one-step and
H5-H20 rollout, and the benefit does not strengthen monotonically with data
scale. The evidence does not justify a Transformer-6L run.

## Source-of-truth artifacts

- `outputs/metrics/world_model_5x3_common_validation.json`: complete progress,
  protocol, all 159 checkpoint records, selections, per-variable metrics.
- `outputs/metrics/world_model_5x3_common_validation_checkpoints.csv`: flat
  candidate table, including selection status and per-variable columns.
- `outputs/metrics/world_model_5x3_common_validation_models.csv`: 15 selected
  architecture checkpoints.
- `outputs/metrics/world_model_5x3_common_validation_best_world_models.csv`:
  three BestWM rows and eligibility thresholds.
- `outputs/metrics/world_model_5x3_common_validation_transformer_depth.csv`:
  2L/4L values and signed depth deltas.
