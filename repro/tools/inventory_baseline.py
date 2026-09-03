#!/usr/bin/env python3
"""Create a drift-detection inventory of committed baseline artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


RUN_TO_K = {
    "20260508_002312": 3,
    "20260507_131506": 5,
    "20260508_104147": 7,
    "20260509_112437": 9,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for run, k in sorted(RUN_TO_K.items(), key=lambda item: item[1]):
        run_dir = args.repository / "Results_HAM_DAA" / run
        for path in sorted(candidate for candidate in run_dir.rglob("*") if candidate.is_file()):
            rows.append(
                {
                    "K": k,
                    "run_directory": run,
                    "run_timestamp_from_directory": (
                        f"{run[:4]}-{run[4:6]}-{run[6:8]}T{run[9:11]}:{run[11:13]}:{run[13:15]}"
                    ),
                    "relative_path": path.relative_to(args.repository).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "is_checkpoint": path.name.endswith(".weights.h5"),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, lineterminator="\n")
    print(f"Inventoried {len(rows)} files ({sum(row['bytes'] for row in rows)} bytes)")


if __name__ == "__main__":
    main()
