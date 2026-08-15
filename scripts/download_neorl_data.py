from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NEORL_ROOT = ROOT / "external" / "NeoRL"
REQUIRED_KEYS = ("obs", "next_obs", "action", "reward", "done", "index")


def _save_dataset(path: Path, data: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"reuse {path}")
        return
    missing = set(REQUIRED_KEYS) - set(data)
    if missing:
        raise ValueError(f"NeoRL dataset is missing keys: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: data[key] for key in REQUIRED_KEYS})
    temporary.replace(path)
    print(f"saved {path}")


def download(scale: int, overwrite: bool = False) -> None:
    if not NEORL_ROOT.exists():
        raise FileNotFoundError(
            "NeoRL source not found. Clone it to external/NeoRL before downloading data."
        )
    sys.path.insert(0, str(NEORL_ROOT))
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    import neorl

    env = neorl.make("ib")
    download_cache = ROOT / "data" / "raw" / "neorl_downloads"
    train, validation = env.get_dataset(
        data_type="medium",
        train_num=scale,
        need_val=True,
        val_ratio=0.1,
        path=str(download_cache),
    )
    _save_dataset(
        ROOT / "data" / "raw" / f"ib-medium-{scale}-train.npz",
        train,
        overwrite,
    )
    _save_dataset(
        ROOT / "data" / "raw" / f"ib-medium-{scale // 10}-val.npz",
        validation,
        overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and export a NeoRL IB Medium dataset scale"
    )
    parser.add_argument("--scale", type=int, choices=(100, 1000, 10000), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download(args.scale, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
