#!/usr/bin/env python3
"""Create validation-learned alignment artifacts without changing raw runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from research.code.alignment import align_coefficients


def configuration(run_dir: Path) -> dict:
    return json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))


def require_completed_primary(run_dir: Path) -> None:
    from research.code.recovery import require_analysis_run
    require_analysis_run(REPOSITORY, run_dir)
    if configuration(run_dir).get("status") != "PRIMARY RUN":
        raise ValueError(f"Alignment accepts completed primary runs only: {run_dir}")


def require_analysis_output(output_dir: Path) -> None:
    root = (REPOSITORY / "research" / "analysis" / "stability_v1").resolve()
    if output_dir == root or not output_dir.is_relative_to(root):
        raise ValueError(f"Alignment output must be a unique child below {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reference_run = args.reference_run.resolve()
    candidate_run = args.candidate_run.resolve()
    output_dir = args.output_dir.resolve()
    require_completed_primary(reference_run)
    require_completed_primary(candidate_run)
    require_analysis_output(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite alignment output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_config = configuration(reference_run)
    cand_config = configuration(candidate_run)
    if ref_config["seed"] != 113:
        raise ValueError("stability_v1 reference seed must be 113")
    if cand_config["seed"] == 113:
        raise ValueError("Candidate must be one of the four non-reference seeds")
    if ref_config["K"] != cand_config["K"]:
        raise ValueError("Reference and candidate K differ")
    ref_mapping = pd.read_csv(reference_run / "validation_mapping.csv")
    cand_mapping = pd.read_csv(candidate_run / "validation_mapping.csv")
    if ref_mapping["image_id"].tolist() != cand_mapping["image_id"].tolist():
        raise ValueError("Validation image order differs between runs")
    if ref_mapping[["lesion_id", "dx"]].astype(str).values.tolist() != cand_mapping[["lesion_id", "dx"]].astype(str).values.tolist():
        raise ValueError("Validation lesion/diagnosis mapping differs between runs")
    reference = np.load(reference_run / "validation_coefficients.npy", allow_pickle=False)
    candidate = np.load(candidate_run / "validation_coefficients.npy", allow_pickle=False)
    result = align_coefficients(reference, candidate)

    pd.DataFrame(result.cost_matrix).to_csv(
        output_dir / "cost_matrix.csv", index=False, header=False, lineterminator="\n"
    )
    pd.DataFrame(result.cost_method_matrix).to_csv(
        output_dir / "cost_method_matrix.csv", index=False, header=False, lineterminator="\n"
    )
    np.save(
        output_dir / "candidate_validation_coefficients_aligned.npy",
        result.aligned_candidate,
        allow_pickle=False,
    )
    test_candidate = np.load(candidate_run / "test_coefficients.npy", allow_pickle=False)
    np.save(
        output_dir / "candidate_test_coefficients_aligned.npy",
        test_candidate[:, result.candidate_column_for_reference],
        allow_pickle=False,
    )
    payload = {
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "K": ref_config["K"],
        "reference_seed": 113,
        "candidate_seed": cand_config["seed"],
        "matching_partition": "validation",
        "test_used_for_matching": False,
        **result.summary(),
    }
    (output_dir / "permutation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
