# PAI Industrial World Model

This repository implements the three-part assignment in `PAI世界模型测试题目.docx`:

1. Temporal Transformer world-model construction on NeoRL Industrial Benchmark data.
2. Strategy optimization against a frozen world model.
3. Simulator evaluation of the original behavior reference, official NeoRL BC, and the world-model-optimized policy.

The current implementation starts with the verified IB Medium 100-trajectory pipeline. Source, environment, and data findings are recorded under `docs/`.

## Verified environment

```powershell
G:\PAI\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "G:\PAI\external\NeoRL"
```

The environment reuses the existing CUDA-enabled packages from Anaconda `base` through `--system-site-packages`; project-specific Gym and NeoRL packages are isolated in `.venv`.

## Train the world model

```powershell
G:\PAI\.venv\Scripts\python.exe scripts\train_world_model.py --config configs\world_model_ib_m100.yaml
```

## Evaluate predictions

```powershell
G:\PAI\.venv\Scripts\python.exe scripts\evaluate_world_model.py --config configs\world_model_ib_m100.yaml
```

The training script writes the best checkpoint and training curve. The evaluation script writes one-step metrics, multi-step metrics, and the required figures.

## Authoritative external sources

- NeoRL main repository: `https://github.com/Polixir/NeoRL`, fixed locally at `717c9a92d5253876f8cb28318ef72e3d5ab05968`.
- NeoRL OfflineRL submodule: fixed at `807933a87f77529f17bd81ac64d717aad89f5cdf`.
- Local task and paper files remain unchanged in the repository root.

