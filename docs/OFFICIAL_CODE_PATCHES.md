# Official Code Compatibility

No third-party source files have been modified.

## Runtime compatibility layer

1. NeoRL's IB implementation uses the removed NumPy alias `np.float`. Project runtime entry points define `np.float = float` before importing NeoRL.
2. NeoRL's wheel omits `data_map.json`. Dataset download and environment work use the fixed official source checkout through `PYTHONPATH=G:\PAI\external\NeoRL`.
3. The Agit OfflineRL submodule endpoint timed out. The exact gitlink commit was fetched from the official `https://github.com/Polixir/OfflineRL` mirror and checked out at the unchanged commit hash.

These changes affect loading/compatibility only; environment dynamics, reward, datasets, and benchmark algorithm code remain unchanged.

