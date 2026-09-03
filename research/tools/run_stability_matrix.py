#!/usr/bin/env python3
"""Print the frozen ten-run matrix; execute only with explicit --execute."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY / "research" / "runs" / "stability_v1",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runner = REPOSITORY / "research" / "tools" / "run_stability_one.py"
    commands: list[list[str]] = []
    for k in (5, 7):
        for seed in (113, 211, 307, 401, 503):
            commands.append(
                [
                    sys.executable,
                    str(runner),
                    "--k", str(k),
                    "--seed", str(seed),
                    "--output-dir", str(args.output_root / f"K{k}" / f"seed_{seed}"),
                    "--image-dirs", *[str(path) for path in args.image_dirs],
                ]
            )
    for command in commands:
        print(subprocess.list2cmdline(command))
    if not args.execute:
        print("DRY RUN ONLY: pass --execute explicitly to launch the frozen matrix.")
        return
    for command in commands:
        subprocess.run(command, cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()

