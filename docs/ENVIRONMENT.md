# Environment Audit

Audit date: 2026-08-14

## Hardware and CUDA layers

- NVIDIA driver: 560.94.
- Driver-reported CUDA capability: 12.6.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM.
- System `nvcc`: not installed / not on PATH.
- Verified PyTorch runtime: PyTorch `2.5.1+cu121`, runtime CUDA `12.1`, `torch.cuda.is_available() == True`.
- No system CUDA Toolkit was installed because the project does not compile CUDA extensions.

## Existing environments

- `base`: Python 3.11.11, CUDA-enabled PyTorch 2.5.1+cu121; selected as the proven package base.
- `d2l`: Python 3.9.18, PyTorch 1.12.0 CPU-only.
- `deepfake`: Python 3.14.3, PyTorch 2.10.0 CPU-only.
- `G:\杂项\package\.conda`: Python 3.11.14, no relevant ML packages detected.

## Isolation decision

An attempted full Conda clone exceeded the five-minute execution window and was automatically cleaned without creating an environment. To avoid modifying `base` or duplicating several GB, the project uses:

```powershell
D:\Program\Anaconda3\python.exe -m venv --system-site-packages G:\PAI\.venv
```

The venv reuses the proven CUDA PyTorch from `base` and contains project-local `gym==0.26.2` plus a local NeoRL wheel. For dataset access, use the official source tree because the NeoRL wheel omits `data_map.json`:

```powershell
G:\PAI\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "G:\PAI\external\NeoRL"
```

## Verified smoke

- `import torch`, `import gym`, and source-tree `import neorl` succeed.
- `neorl.make("ib")` succeeds.
- `reset()` returns `(180,)`.
- `step()` returns `(180,)`, scalar reward, `done=False`, and info.
- `get_dataset(data_type="medium", train_num=100, path="G:\PAI\data\raw")` succeeds.
- GPU name and CUDA availability were verified from the project venv.

## Compatibility notes

- Gym 0.26 prints its upstream maintenance notice; NeoRL uses the legacy four-return `step()` API internally and still runs.
- Official IB code references removed NumPy alias `np.float`. Project entry points install the runtime alias `np.float = float` before importing/running NeoRL; third-party source is unchanged.
- NeoRL 0.3.0 `setup.py` does not package `data_map.json`. Dataset operations therefore use the official source checkout via `PYTHONPATH`.

