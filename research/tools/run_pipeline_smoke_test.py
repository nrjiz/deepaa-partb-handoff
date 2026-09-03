#!/usr/bin/env python3
"""One-batch, one-epoch non-scientific pipeline smoke test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from research.code.determinism import process_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    seed = 113
    required_environment = process_environment(seed)
    if any(os.environ.get(key) != value for key, value in required_environment.items()):
        environment = os.environ.copy()
        environment.update(required_environment)
        subprocess.run([sys.executable, *sys.argv], env=environment, check=True)
        return

    from research.code.runner import RunSpec, execute_run

    split = REPOSITORY / "research" / "splits" / "journal_v1"
    run_dir = execute_run(
        RunSpec(
            repository=REPOSITORY,
            k=5,
            seed=seed,
            output_dir=args.output_dir,
            train_manifest=split / "train.csv",
            validation_manifest=split / "validation.csv",
            test_manifest=split / "test.csv",
            image_dirs=args.image_dirs,
            smoke_test=True,
            smoke_rows_per_partition=32,
            smoke_epochs=1,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
