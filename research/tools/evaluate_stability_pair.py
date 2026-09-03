#!/usr/bin/env python3
"""Evaluate one aligned reference/candidate pair on held-out test artifacts."""

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

from research.code.constants import EXPECTED_CLASSES
from research.code.model_adapter import load_baseline_module
from research.code.stability_metrics import (
    coefficient_stability,
    lesion_bootstrap_coefficient_stability,
    nearest_real_top_n,
    paired_lesion_coefficients,
    compare_nearest_real,
    semantic_stability,
    visual_stability,
)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def require_analysis_output(output_dir: Path) -> None:
    root = (REPOSITORY / "research" / "analysis" / "stability_v1").resolve()
    if output_dir == root or not output_dir.is_relative_to(root):
        raise ValueError(f"Stability output must be a unique child below {root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reference_run = args.reference_run.resolve()
    candidate_run = args.candidate_run.resolve()
    alignment_dir = args.alignment_dir.resolve()
    output_dir = args.output_dir.resolve()
    require_analysis_output(output_dir)
    for run_dir in (reference_run, candidate_run):
        if not (run_dir / "COMPLETED.json").is_file():
            raise ValueError(f"Run lacks COMPLETED.json: {run_dir}")
        run_config = json.loads((run_dir / "configuration.json").read_text(encoding="utf-8"))
        if run_config.get("status") != "PRIMARY RUN":
            raise ValueError(f"Evaluation accepts completed primary runs only: {run_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite stability output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    permutation_record = json.loads((alignment_dir / "permutation.json").read_text(encoding="utf-8"))
    if Path(permutation_record["reference_run"]).resolve() != reference_run:
        raise ValueError("Alignment reference run does not match the requested reference run")
    if Path(permutation_record["candidate_run"]).resolve() != candidate_run:
        raise ValueError("Alignment candidate run does not match the requested candidate run")
    if permutation_record.get("matching_partition") != "validation" or permutation_record.get("test_used_for_matching") is not False:
        raise ValueError("Alignment record does not prove validation-only matching")
    permutation = np.asarray(permutation_record["candidate_column_for_reference"], dtype=int)
    reference_coefficients = np.load(reference_run / "test_coefficients.npy", allow_pickle=False)
    candidate_raw = np.load(candidate_run / "test_coefficients.npy", allow_pickle=False)
    if reference_coefficients.shape[1] != permutation_record["K"]:
        raise ValueError("Alignment K does not match test coefficient K")
    candidate_aligned = candidate_raw[:, permutation]
    saved_candidate_aligned = np.load(
        alignment_dir / "candidate_test_coefficients_aligned.npy", allow_pickle=False
    )
    np.testing.assert_array_equal(candidate_aligned, saved_candidate_aligned)
    ref_mapping = pd.read_csv(reference_run / "test_mapping.csv")
    cand_mapping = pd.read_csv(candidate_run / "test_mapping.csv")
    if ref_mapping["image_id"].tolist() != cand_mapping["image_id"].tolist():
        raise ValueError("Test image order differs between runs")
    if ref_mapping[["lesion_id", "dx"]].astype(str).values.tolist() != cand_mapping[["lesion_id", "dx"]].astype(str).values.tolist():
        raise ValueError("Test lesion/diagnosis mapping differs between runs")

    coefficient = coefficient_stability(
        reference_coefficients, ref_mapping, candidate_aligned, cand_mapping
    )
    ref_lesion, cand_lesion, lesion_mapping = paired_lesion_coefficients(
        reference_coefficients, ref_mapping, candidate_aligned, cand_mapping
    )
    bootstrap = lesion_bootstrap_coefficient_stability(ref_lesion, cand_lesion)
    semantic = semantic_stability(
        reference_coefficients,
        ref_mapping["dx"],
        candidate_aligned,
        cand_mapping["dx"],
    )
    baseline = load_baseline_module(REPOSITORY)
    k = reference_coefficients.shape[1]
    z_fixed = baseline.create_z_fixed(k - 1)
    ref_nearest = nearest_real_top_n(
        np.load(reference_run / "test_latent_means.npy", allow_pickle=False),
        z_fixed,
        ref_mapping["image_id"],
        ref_mapping["dx"],
    )
    cand_nearest = nearest_real_top_n(
        np.load(candidate_run / "test_latent_means.npy", allow_pickle=False),
        z_fixed,
        cand_mapping["image_id"],
        cand_mapping["dx"],
    )
    nearest = compare_nearest_real(ref_nearest, cand_nearest, permutation)
    visual = visual_stability(
        np.load(reference_run / "decoded_archetypes_raw.npy", allow_pickle=False),
        np.load(candidate_run / "decoded_archetypes_raw.npy", allow_pickle=False),
        permutation,
    )

    lesion_rows = lesion_mapping.copy()
    lesion_rows["coefficient_vector_l1"] = coefficient["coefficient_vector_l1_by_lesion"]
    lesion_rows["dominant_archetype_agreement"] = coefficient["dominant_agreement_by_lesion"]
    lesion_rows.to_csv(
        output_dir / "coefficient_stability_by_lesion.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(nearest).to_csv(
        output_dir / "nearest_real_stability.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(visual).to_csv(
        output_dir / "visual_stability.csv", index=False, lineterminator="\n"
    )
    semantic_rows = []
    for archetype in range(k):
        for class_index, diagnosis in enumerate(EXPECTED_CLASSES):
            semantic_rows.append(
                {
                    "matched_archetype": archetype + 1,
                    "dx": diagnosis,
                    "reference_probability": semantic["reference_distribution"][archetype, class_index],
                    "candidate_probability": semantic["candidate_distribution"][archetype, class_index],
                    "jensen_shannon_divergence_base2": semantic["jensen_shannon_divergence_base2"][archetype],
                }
            )
    pd.DataFrame(semantic_rows).to_csv(
        output_dir / "semantic_stability.csv", index=False, lineterminator="\n"
    )
    summary = {
        "matching_partition": "validation",
        "evaluation_partition": "test",
        "reference_seed": 113,
        "candidate_seed": permutation_record["candidate_seed"],
        "K": k,
        "permutation": permutation.tolist(),
        "coefficient": {
            key: value for key, value in coefficient.items()
            if key not in {"coefficient_vector_l1_by_lesion", "dominant_agreement_by_lesion", "lesion_mapping"}
        },
        "semantic": {
            "mean_jensen_shannon_divergence_base2": semantic["mean_jensen_shannon_divergence_base2"]
        },
        "nearest_real": nearest,
        "visual": visual,
        "lesion_bootstrap": bootstrap,
    }
    (output_dir / "stability_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
