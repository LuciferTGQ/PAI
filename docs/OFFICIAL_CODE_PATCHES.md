# Official Code Compatibility

No third-party source files have been modified.

## Runtime compatibility layer

1. NeoRL's IB implementation uses the removed NumPy alias `np.float`. Project runtime entry points define `np.float = float` before importing NeoRL.
2. NeoRL's wheel omits `data_map.json`. Dataset download and environment work use the fixed official source checkout through `PYTHONPATH=G:\PAI\external\NeoRL`.
3. The Agit OfflineRL submodule endpoint timed out. The exact gitlink commit was fetched from the official `https://github.com/Polixir/OfflineRL` mirror and checked out at the unchanged commit hash.

These changes affect loading/compatibility only; environment dynamics, reward, datasets, and benchmark algorithm code remain unchanged.

## OfflineRL BC compatibility layer

Importing the fixed OfflineRL package executes its top-level experiment stack
and currently fails first on the unrelated legacy `aim` dependency. Module B
therefore uses `src/strategy/official_bc.py`, a narrow compatibility layer for
the audited BC path only. It preserves the fixed commit's GaussianActor
architecture, LeakyReLU slope, learnable soft log-standard-deviation bounds,
Gaussian negative-log-likelihood update, unbounded deterministic mean inference,
and summed validation batch-MSE model selection.

The official config values are kept in
`configs/official_bc_ib_m100.yaml`: hidden size 256, two hidden layers, batch
size 256, 1,000 steps per epoch, 100 epochs, and Adam learning rate 1e-3.
This implements the code release, whose learning rate differs from the paper's
3e-4 statement; the discrepancy remains recorded in `SOURCE_AUDIT.md`.
