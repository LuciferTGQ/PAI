from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def extract_npz_arrays(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise ValueError(f"Expected an existing .npz file, got {source}")
    destination.mkdir(parents=True, exist_ok=True)
    required = {f"{key}.npy" for key in ("obs", "next_obs", "action", "reward", "done", "index")}
    with zipfile.ZipFile(source) as archive:
        members = {member.filename: member for member in archive.infolist()}
        missing = required - set(members)
        if missing:
            raise ValueError(f"Source archive is missing {sorted(missing)}")
        for filename in sorted(required):
            member = members[filename]
            output = destination / filename
            if output.exists() and output.stat().st_size == member.file_size:
                print(f"reuse {output} ({member.file_size} bytes)")
                continue
            temporary = destination / f"{filename}.partial"
            with archive.open(member) as input_handle, temporary.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=16 * 1024 * 1024)
            if temporary.stat().st_size != member.file_size:
                raise IOError(f"Incomplete extraction for {filename}")
            temporary.replace(output)
            print(f"extracted {output} ({member.file_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compressed NPZ arrays for mmap training")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    extract_npz_arrays(args.source, args.destination)


if __name__ == "__main__":
    main()
