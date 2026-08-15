# Figure contracts

All plotting uses Python/matplotlib exclusively. Final quantitative figures are double-column compatible (approximately 178 mm wide), use editable vector text, and export PDF/SVG plus 600 dpi PNG. Error bars denote population standard deviation over simulator seeds unless stated otherwise.

## Figure 1

Core conclusion: Dynamics validation selects a frozen World Model before model-based planning or reinforcement learning is evaluated in the simulator.

Evidence hierarchy: workflow separation is the hero evidence; the validation-only gate is the key control.

Reviewer risk: the diagram must not imply that simulator reward selects the World Model.

## Figure 2

Core conclusion: World Model accuracy depends jointly on architecture, data scale, and prediction horizon.

Evidence hierarchy: annotated one-step and H5/H10/H20 heatmaps are equal primary evidence.

Reviewer risk: all cells must come from the same common-validation protocol.

## Figure 3

Core conclusion: Similar one-step error can lead to different recursive error accumulation.

Evidence hierarchy: the five measured horizons are primary evidence; connecting segments are only visual guides.

Reviewer risk: categorical horizon positions must not be interpreted as interpolated continuous measurements.

## Figure 4

Core conclusion: Recursive prediction drift is variable-specific and becomes visible within an H50 rollout.

Evidence hierarchy: ground truth versus prediction for velocity, fatigue, and consumption.

Reviewer risk: one representative trajectory cannot establish population-level accuracy; the aggregate NRMSE results remain primary.

## Figure 5

Core conclusion: Strategy ranking changes with World Model data scale, and strong optimizers can fail on weak models.

Evidence hierarchy: individual matched development seeds plus mean and standard deviation.

Reviewer risk: the off-scale M100 iCEM result must remain visible and numerically labeled.

## Figure 6

Core conclusion: Dynamics-validation ranking is not identical to downstream ranking and differs between planning and learned-policy use.

Evidence hierarchy: MPPI and MB-PPO controlled panels; identical seeds and within-panel protocols.

Reviewer risk: this five-architecture observation must not be generalized as a universal law.

## Figure 7

Core conclusion: The final selected systems improve cumulative reward over BC on untouched simulator seeds.

Evidence hierarchy: ten online seed dots and mean/standard deviation; Original Behavior is visually separated as an unpaired dataset-trajectory distribution.

Reviewer risk: Original Behavior must not be presented as matched-seed online evaluation.

## Figure 8

Core conclusion: Behavior KL limits policy divergence and prevents severe MB-PPO return collapse in the observed ablation.

Evidence hierarchy: simulator-return dots plus dense gradient-step KL history.

Reviewer risk: one World Model and one training seed support an ablation observation, not a universal guarantee.
