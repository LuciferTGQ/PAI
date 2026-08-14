# Data Audit

Audit date: 2026-08-14

Dataset: NeoRL Industrial Benchmark, Medium, official 100-trajectory training file and 10-trajectory validation file.

## Verified files

| File | Bytes | Official MD5 verified |
|---|---:|---|
| `ib-medium-100-train.npz` | 5,365,472 | `afd8973406bc7401ce77d9b0b4695947` |
| `ib-medium-10-val.npz` | 534,897 | `eb146a26eef350ad116901bc36a1d833` |
| `ib-medium-1000-train.npz` | 53,673,961 | `8aeededf5c3baf3eed30f6ca8080fe02` |
| `ib-medium-100-val.npz` | 5,333,465 | `27839d327ac8a933a0d0a91ed567935b` |
| `ib-medium-10000-train.npz` | 536,633,591 | `7f601ab6adae6790635801812aa66b7f` |
| `ib-medium-1000-val.npz` | 53,338,808 | `deed6007666f35af468cc22d9026bb1d` |

The 1000/10000 files are cached only. Formal training remains gated on completion of the IB-M-100 pipeline.

## Shapes and dtypes

| Split | Field | Shape | dtype |
|---|---|---|---|
| train | `obs` | `(100000, 180)` | `float32` |
| train | `next_obs` | `(100000, 180)` | `float32` |
| train | `action` | `(100000, 3)` | `float32` |
| train | `reward` | `(100000, 1)` | `float32` |
| train | `done` | `(100000, 1)` | `int64` |
| train | `index` | `(100,)` | `int64` |
| validation | `obs` | `(10000, 180)` | `float32` |
| validation | `next_obs` | `(10000, 180)` | `float32` |
| validation | `action` | `(10000, 3)` | `float32` |
| validation | `reward` | `(10000, 1)` | `float32` |
| validation | `done` | `(10000, 1)` | `bool` |
| validation | API `index` | `(10,)` | `float64` |

## Trajectory and boundary semantics

- Training: 100 trajectories, 1000 transitions each, 100 terminal flags at indices 999, 1999, ..., 99999.
- Validation: 10 trajectories, 1000 transitions each, 10 terminal flags.
- Raw `.npz` `index` values are exclusive trajectory ends (`1000, 2000, ..., N`). NeoRL `get_dataset()` inserts 0 and returns start indices after slicing.
- Project loading code normalizes both representations into explicit `(start, end)` spans.

## 30 x 6 structure

- Observation is exactly 30 frames x 6 variables, flattened latest frame first.
- Variable order is `[setpoint, velocity, gain, shift, fatigue, consumption]`.
- For every one of the 100,000 training transitions, `next_obs[:, 6:]` exactly equals `obs[:, :-6]`.
- The model therefore predicts only the six-dimensional next frame and reconstructs the remaining 29 frames deterministically.

## Values and reward

- Action component bounds in the sampled training data are within `[-1, 1]`; simulator bounds are exactly `[-1, 1]^3`.
- Training reward range: `[-374.59235, -135.92253]`; mean `-282.82217`.
- Validation reward range: `[-343.96274, -155.05344]`; mean `-283.71762`.
- Recomputed reward `-(3 * next_fatigue + next_consumption)` matches stored reward within float32 rounding.

## Leakage controls

- Normalization statistics are computed from training data only.
- Validation trajectories are never included in normalization or parameter fitting.
- Multi-step rollout starts and action sequences are constrained to a single validation trajectory; true states are not re-injected after the initial history.

