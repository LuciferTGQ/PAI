# Major Decisions

## 2026-08-14 - World-model state definition

- Evidence: released IB data and current simulator are 180-dimensional, organized as 30 latest-to-oldest frames of six variables.
- Decision: the Transformer consumes frames in chronological oldest-to-latest order after reversing the flattened source representation, and predicts the next six-dimensional frame directly.
- Reconstruction: prepend the predicted frame to the original latest-to-oldest flattened history after dropping the oldest frame.
- Reason: the other 29 frames are deterministic copies and should not consume model capacity.

## 2026-08-14 - Environment isolation

- Decision: use a project `.venv` with `--system-site-packages` over the verified CUDA-enabled Anaconda `base`.
- Reason: preserve the user's working GPU environment while avoiding a large full clone and any CUDA Toolkit download.

## 2026-08-14 - Official BC reference

- Decision: target fixed OfflineRL commit `807933a87f77529f17bd81ac64d717aad89f5cdf`, not the evolving current upstream API.
- Known source conflict: paper learning rate is `3e-4`; fixed code config uses `1e-3`. Compatibility implementation will default to fixed code and document deviations.

## 2026-08-14 - Original behavior comparison

- Decision: unless an official IB SAC checkpoint is found, use complete dataset trajectory returns as `Original Behavior Policy - empirical dataset performance`.
- This value must not be described as an online replay result.

