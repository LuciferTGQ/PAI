# M-100 Smoke Results

These results validate the complete pipeline on the released IB Medium
100-trajectory training split. They are smoke results, not the planned M-1000
formal experiment.

## Temporal Transformer

The best five-epoch validation loss was 0.039066 in normalized target space.
Fresh-process checkpoint evaluation on 10,000 validation transitions produced:

- one-step MSE 2.7833, MAE 0.7248, RMSE 1.6683, and std-normalized RMSE 0.1976;
- free-running MSE 3.1390 at horizon 1, 8.0669 at horizon 5, 19.4754 at
  horizon 10, 22.3052 at horizon 20, and 35.5853 at horizon 50.

The rollout test used the true future action sequence, never reinserted true
states, and never crossed an episode boundary.

## Official-code-compatible BC

The actor used the fixed OfflineRL code configuration: two 256-unit hidden
layers, batch size 256, Adam 1e-3, and 100 epochs of 1,000 sampled updates.
Training became numerically unstable after epoch 34, recovered, and diverged
again. The official validation-selection rule correctly retained epoch 88,
whose action MSE was 0.10309. This instability is preserved as an observed
result rather than hidden by changing official hyperparameters.

## Simulator policy comparison

All online policies ran for 1,000 steps on official IBGym with matched seeds
42, 43, and 44. Higher (less negative) return is better.

| Policy | Evaluation type | Mean return | Std |
|---|---|---:|---:|
| Released behavior trajectories | Offline reference, 100 episodes | -282,822.15 | 1,265.64 |
| Official BC | Online, 3 matched seeds | -286,560.77 | 122.71 |
| Frozen-world-model CEM | Online, 3 matched seeds | -278,320.74 | 307.40 |

WM-CEM improved over BC by 8,240.03 mean return across the matched online
seeds. Its per-seed improvements were all positive. It also exceeded the
offline behavior-data mean by 4,501.41, but that comparison is descriptive
only because the released trajectories do not share the online seeds.

The official BC mean is unbounded by design; 2,992 of 9,000 scalar actions
(33.24%) were clipped only at the simulator boundary. WM-CEM searches within
the legal action box and required no boundary clipping.

The CEM smoke configuration used horizon 5, population 64, 8 elites, and two
iterations. The Temporal Transformer checkpoint remained frozen throughout.
