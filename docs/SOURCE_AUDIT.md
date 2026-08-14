# Source Audit

Audit date: 2026-08-14

## Sources and precedence

1. `[TASK]` `G:\PAI\PAI世界模型测试题目.docx` (4 pages; text extracted and visually checked).
2. `[PAPER]` `G:\PAI\2102.00714v2.pdf` (NeoRL v2, 29 pages; text extracted; figures/tables checked where material).
3. `[OFFICIAL CODE]` `https://github.com/Polixir/NeoRL`, local commit `717c9a92d5253876f8cb28318ef72e3d5ab05968`.
4. `[OFFICIAL CODE]` NeoRL `benchmark/OfflineRL` submodule, fixed commit `807933a87f77529f17bd81ac64d717aad89f5cdf` (retrieved from the official Polixir GitHub mirror after the Agit endpoint timed out).

## Assignment requirements

- `[TASK]` Learn a state-transition/world model from historical `(state, action, next state, reward)` trajectories.
- `[TASK]` Report one-step prediction accuracy and multi-step error accumulation.
- `[TASK]` optimize a control policy using the learned world model.
- `[TASK]` deploy policies in the NeoRL industrial simulator and compare original behavior, a basic policy, and a world-model-optimized policy using episode reward.
- `[TASK]` deliver source code and a concise experiment report; the task permits MLP, recurrent, Transformer, or other temporal models.

## NeoRL data construction

- `[PAPER]` NeoRL targets near-real-world offline RL: conservative data, limited data, stochastic dynamics, and offline validation before deployment.
- `[PAPER]` SAC is trained to convergence. The highest-return checkpoint is called expert; policies near 25%, 50%, and 75% of expert return define low, medium, and high quality.
- `[PAPER]` Four similar policies are selected per level. Three generate training data and the remaining one generates validation/test data. Validation size is one tenth of training size.
- `[PAPER]` Generic environments inject action noise with probability 20%; IB does not add explicit action noise because its dynamics are already highly stochastic. The IB behavior policy is therefore deterministic.
- `[PAPER]` The paper reports 99/999/9999 trajectories. `[OFFICIAL CODE]` the released API accepts requested counts below 10,000 and maps them to official 100/1000/10000 training files plus 10/100/1000 validation files. The benchmark configuration passes 99, but the assignment and this implementation use the released 100-trajectory file explicitly.
- `[OFFICIAL CODE]` Dataset dictionaries contain `obs`, `next_obs`, `action`, `reward`, `done`, and `index`.

## Industrial Benchmark semantics

- `[PAPER]` Each raw system output frame is `[setpoint, velocity, gain, shift, fatigue, consumption]`.
- `[PAPER]` The appendix states an IB observation shape of 182: 30 six-variable frames plus two dummy reward-related values.
- `[OFFICIAL CODE]` the current `IBGym` uses `include_past`, `n_past_timesteps=30`, and exposes a 180-dimensional Box. A new frame is inserted at index 0 and the oldest frame is removed, so flattened observations are latest-to-oldest.
- `[VERIFIED LOCALLY]` the simulator and released IB Medium data are both 180-dimensional, not 182-dimensional. Every released transition satisfies `next_obs[:, 6:] == obs[:, :-6]`.
- `[OFFICIAL CODE]` action shape is 3 and each continuous steering change is bounded to `[-1, 1]`.
- `[OFFICIAL CODE]` episode length is 1000 steps and the environment has no early terminal condition.
- `[OFFICIAL CODE]` reward is `-(3 * next_fatigue + next_consumption)`. `[VERIFIED LOCALLY]` this reproduces released rewards within float32 rounding (`max abs error 3.0517578e-05`).

## Official baselines

- `[PAPER]` evaluated baselines include BC, CQL, PLAS, BCQ, MOPO, and MB-PPO, together with expert, deterministic behavior, behavior, and random references.
- `[OFFICIAL CODE]` the fixed OfflineRL commit contains BC, BCQ, CQL, PLAS, CRR, BREMEN, MOPO, MOReL, and related implementations. The current upstream repository has since added more methods; those are not treated as part of the paper's fixed benchmark.

### Official BC at the fixed benchmark commit

- Actor: `GaussianActor`.
- Architecture: MLP from observation to `2 * action_dim`; two hidden layers, 256 units each, LeakyReLU(0.1). Output is split into Gaussian mean and log standard deviation; log std is softly clamped between learned lower/upper parameters.
- Training loss: negative action log likelihood under the Gaussian.
- Optimizer: Adam.
- Code config: actor learning rate `1e-3`, batch size 256, 1000 steps per epoch, 100 epochs (100K gradient steps), seed 42.
- Paper appendix: learning rate `3e-4`, batch size 256, 100K steps, early stopping by lowest validation NLL.
- Fixed code behavior: model selection uses summed validation MSE between Gaussian mean and action; returned policy is the best copied actor. `policy_infer` returns the unbounded Gaussian mean; the BC implementation does not apply tanh or clipping in this path.
- `[ENGINEERING CHOICE]` integration must preserve the fixed code behavior by default and clearly expose any safety clipping applied only at simulator action submission.

## Original behavior policy checkpoint

- `[PAPER]` behavior policies were SAC policies used for data collection.
- `[OFFICIAL CODE]` neither the fixed NeoRL main repository nor the fixed OfflineRL submodule contains the original IB SAC checkpoint or a loader for it; no model/checkpoint artifacts are tracked.
- `[UNVERIFIED]` no public checkpoint was found in the audited official repositories. Until an official artifact is located, the final comparison must use empirical dataset trajectory return as the original-behavior reference and label it as offline empirical performance, not online replay.

## Source conflicts resolved

- Runtime/data shape uses 180 dimensions because both current official source and released data agree; the paper's 182-dimensional appendix statement is retained as historical context.
- Official BC compatibility targets the fixed code commit. The paper/code learning-rate and early-stopping differences remain explicit rather than silently merged.

