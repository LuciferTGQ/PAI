# iCEM and MPPI Reference Audit

## iCEM

Source of truth: `martius-lab/iCEM` commit
`98c1c1fe2bfc94a87e4658b6c793f4e81bc30203`, controller
`icem/controllers/icem.py`.

The corrected implementation now uses the official `colorednoise` dependency,
action-bound midpoint initialization, iterative integer population decay,
clipping, elite reuse, previous-elite shift, final-iteration mean sample, and
best evaluated sequence execution. The old custom noise generator was not
equivalent because it removed the DC component and normalized each short
realization independently.

The deterministic toy objective passed: MSE decreased from 0.422500 to
0.001262. The generated colored noise is exactly equal to the official
dependency call for an identical random generator.

Development results:

| World Model | Frozen iCEM setting | Mean return | Std | Median |
|---|---|---:|---:|---:|
| BestWM@M1000, Transformer-2L e20 | beta2, population40 | -334596.78 | 4503.00 | -332570.07 |
| BestWM@M10000, Transformer-2L e4 | beta0, population40 | -321264.07 | 36310.26 | -317320.62 |

For M-10000, beta2 predicted better model returns while producing much worse
simulator returns and saturated actions. White noise restored control across
all five development seeds. Beta3 and a doubled sampling budget did not fix
the failure. This is a noise/model-exploitation interaction, not evidence that
iCEM is inherently unsuitable for IB.

## MPPI

Source of truth: `UM-ARM-Lab/pytorch_mppi` v0.9.1, commit
`e04a569cd4215e705f9013145c496fc59cb25ed6`. The package core is unmodified.
Only FrozenWorldModel dynamics, the IB running cost, bounds, and the policy
interface are adapted.

The deterministic integrator toy test passed: planned cost 5.0645 versus
zero-action cost 72.0. The historical custom MPPI remains in
`src/strategy/mppi_mpc.py` for provenance but is not used for formal results.

Development results with H10, 512 samples, noise std 0.15, temperature 100:

| World Model | Mean return | Std | Median | Mean ESS | Runtime/episode |
|---|---:|---:|---:|---:|---:|
| BestWM@M1000, Transformer-2L e20 | -221071.78 | 1475.16 | -221633.10 | 481.83 | 26.95 s |
| BestWM@M10000, Transformer-2L e4 | -223773.80 | 578.22 | -223452.47 | 475.40 | 28.37 s |

The reference implementation is strong and stable while the historical custom
implementation was poor. The previous MPPI failure is therefore classified as
an implementation problem, not an algorithm-level IB conclusion.

## Frozen planner configurations

- iCEM: H10, population40, elites10, three iterations, population decay1.25,
  reuse0.3, alpha0.1; beta2 for M-100/M-1000 and beta0 for M-10000.
- MPPI: H10, 512 samples, diagonal noise std0.15, temperature100.
- Seeds100-109 remain unused.

Raw JSON and timestep CSV files under `outputs/metrics/` are the source of
truth; the main system matrix can be regenerated without rerunning these
diagnostic episodes.
