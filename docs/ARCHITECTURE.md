# Architecture

The project keeps the three assignment modules separate so that later policy
changes cannot silently alter the learned dynamics model.

## Module A: Temporal Transformer world model

- Input state: 30 six-variable IB frames stored newest-first by NeoRL.
- Model tokens: the same frames reordered oldest-first for causal chronology.
- Action input: the current three-dimensional control action.
- Output: the next six-dimensional frame
  `[setpoint, velocity, gain, shift, fatigue, consumption]`.
- State transition: prepend the prediction and discard the oldest frame.
- Normalization: per-variable statistics fitted on the training split only.

`FrozenWorldModel` is the stable boundary exposed to downstream code. It loads
a checkpoint, freezes all parameters, switches the network to evaluation mode,
and provides `predict_next_frame(history, action)` and `rollout(...)`.

## Module B: strategy optimization

This module consumes only the frozen interface. The official NeoRL behavior
cloning policy is the basic-policy baseline. Model-based strategy optimization
will search future action sequences against the frozen world model and the IB
reward definition; it will not update world-model parameters.

## Module C: simulator evaluation

The NeoRL IB environment is the final judge. Evaluation reports:

1. empirical returns of the released behavior trajectories as the original
   behavior reference;
2. online returns of the official-code-compatible BC baseline;
3. online returns of the policy optimized with the frozen world model.

All online comparisons use matched seeds and the same episode horizon. Action
clipping, if needed at the simulator boundary, is recorded explicitly.
