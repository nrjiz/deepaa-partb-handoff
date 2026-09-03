#!/usr/bin/env python3
"""Launch exactly one primary stability_v1 run."""

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


def parse_args() -> argparse.Namespace:
    default_split = REPOSITORY / "research" / "splits" / "journal_v1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, choices=(5, 7))
    parser.add_argument("--seed", type=int, required=True, choices=(113, 211, 307, 401, 503))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=default_split / "train.csv")
    parser.add_argument("--validation-manifest", type=Path, default=default_split / "validation.csv")
    parser.add_argument("--test-manifest", type=Path, default=default_split / "test.csv")
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--allow-completed-rerun",
        action="store_true",
        help="Never overwrites: allocates a timestamped sibling when the requested run is complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_environment = process_environment(args.seed)
    if any(os.environ.get(key) != value for key, value in required_environment.items()):
        environment = os.environ.copy()
        environment.update(required_environment)
        subprocess.run([sys.executable, *sys.argv], env=environment, check=True)
        return

    from research.code.runner import RunSpec, execute_run

    run_dir = execute_run(
        RunSpec(
            repository=REPOSITORY,
            k=args.k,
            seed=args.seed,
            output_dir=args.output_dir,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            test_manifest=args.test_manifest,
            image_dirs=args.image_dirs,
            allow_completed_rerun=args.allow_completed_rerun,
        )
    )
    print(run_dir)


if __name__ == "__main__":
    main()
