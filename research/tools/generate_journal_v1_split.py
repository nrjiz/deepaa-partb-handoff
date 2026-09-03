#!/usr/bin/env python3
"""Generate the predetermined, lesion-grouped HAM10000 journal_v1 split.

This utility is intentionally separate from the historical conference code and
never trains a model. Fold 0 is test, fold 1 is validation, and folds 2-9 are
train. It refuses to write final manifests if lesion labels are inconsistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


EXPECTED_IMAGES = 10_015
EXPECTED_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
PARTITION_ORDER = ["train", "validation", "test"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_inventory(image_dirs: list[Path]) -> tuple[dict[str, str], list[str]]:
    locations: dict[str, str] = {}
    duplicates: list[str] = []
    for image_dir in image_dirs:
        for image_path in image_dir.glob("*.jpg"):
            if image_path.stem in locations:
                duplicates.append(image_path.stem)
            locations[image_path.stem] = image_dir.name
    return locations, sorted(set(duplicates))


def validate_metadata(metadata: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    required = {"image_id", "lesion_id", "dx"}
    missing_columns = required.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {sorted(missing_columns)}")
    if len(metadata) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} metadata rows, found {len(metadata)}")
    if metadata["image_id"].isna().any() or not metadata["image_id"].is_unique:
        raise ValueError("image_id must be present and unique for every row")
    if metadata["lesion_id"].isna().any():
        raise ValueError("lesion_id is missing for one or more rows")
    classes = sorted(metadata["dx"].astype(str).unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected diagnosis classes: {classes}")

    label_counts = metadata.groupby("lesion_id", sort=True)["dx"].nunique(dropna=False)
    conflicting_ids = label_counts[label_counts > 1].index
    conflicts = metadata[metadata["lesion_id"].isin(conflicting_ids)][
        ["lesion_id", "image_id", "dx"]
    ].sort_values(["lesion_id", "image_id"])
    return classes, conflicts


def assign_folds(metadata: pd.DataFrame) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=113)
    fold = np.full(len(metadata), -1, dtype=np.int64)
    placeholder = np.zeros((len(metadata), 1), dtype=np.uint8)
    for fold_number, (_, held_out_indices) in enumerate(
        splitter.split(placeholder, metadata["dx"], groups=metadata["lesion_id"])
    ):
        if np.any(fold[held_out_indices] != -1):
            raise RuntimeError("An image was assigned to more than one fold")
        fold[held_out_indices] = fold_number
    if np.any(fold == -1):
        raise RuntimeError("One or more images were not assigned to a fold")
    return fold


def partition_for_fold(fold: int) -> str:
    if fold == 0:
        return "test"
    if fold == 1:
        return "validation"
    return "train"


def distribution_table(frames: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> pd.DataFrame:
    overall_percent = metadata["dx"].value_counts(normalize=True).mul(100)
    rows: list[dict] = []
    all_frames = {"overall": metadata, **frames}
    for partition, frame in all_frames.items():
        for diagnosis in EXPECTED_CLASSES:
            image_count = int((frame["dx"] == diagnosis).sum())
            image_percent = float(100 * image_count / len(frame))
            rows.append(
                {
                    "partition": partition,
                    "dx": diagnosis,
                    "image_count": image_count,
                    "image_percent": image_percent,
                    "overall_image_percent": float(overall_percent[diagnosis]),
                    "deviation_percentage_points": float(
                        image_percent - overall_percent[diagnosis]
                    ),
                    "lesion_count": int(frame.loc[frame["dx"] == diagnosis, "lesion_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def invariant_results(
    metadata: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    image_locations: dict[str, str],
    duplicate_files: list[str],
) -> list[dict]:
    combined = pd.concat(frames.values(), ignore_index=True)
    metadata_ids = set(metadata["image_id"].astype(str))
    combined_ids = set(combined["image_id"].astype(str))
    available_ids = set(image_locations)
    results: list[dict] = []

    def add(name: str, passed: bool, details: str) -> None:
        results.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})

    add("metadata_row_count", len(metadata) == EXPECTED_IMAGES, f"actual={len(metadata)}")
    add("metadata_image_id_unique", metadata["image_id"].is_unique, f"unique={metadata['image_id'].nunique()}")
    add("every_image_exactly_once", len(combined) == len(metadata) and combined["image_id"].is_unique,
        f"combined_rows={len(combined)}; unique={combined['image_id'].nunique()}")
    add("no_images_lost", combined_ids == metadata_ids,
        f"missing={len(metadata_ids - combined_ids)}")
    add("no_unknown_images_in_manifests", combined_ids <= metadata_ids,
        f"unknown={len(combined_ids - metadata_ids)}")
    add("all_metadata_images_available", metadata_ids <= available_ids,
        f"missing_files={len(metadata_ids - available_ids)}")
    add("no_unknown_available_image_files", available_ids <= metadata_ids,
        f"extra_files={len(available_ids - metadata_ids)}")
    add("no_duplicate_image_files_across_sources", not duplicate_files,
        f"duplicate_file_ids={len(duplicate_files)}")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        image_overlap = set(frames[left]["image_id"]) & set(frames[right]["image_id"])
        lesion_overlap = set(frames[left]["lesion_id"]) & set(frames[right]["lesion_id"])
        add(f"image_ids_disjoint_{left}_{right}", not image_overlap,
            f"overlap={len(image_overlap)}")
        add(f"lesion_ids_disjoint_{left}_{right}", not lesion_overlap,
            f"overlap={len(lesion_overlap)}")
    for partition, frame in frames.items():
        missing_classes = sorted(set(EXPECTED_CLASSES) - set(frame["dx"]))
        add(f"all_classes_represented_{partition}", not missing_classes,
            f"missing={missing_classes}")
    return results


def render_audit(summary: dict) -> str:
    lines = [
        "# journal_v1 leakage and invariant audit",
        "",
        "Status: **PROPOSED JOURNAL SPLIT V1 — pending senior-developer review**.",
        "",
        "The split was generated once using the predetermined rule. No alternative seeds or fold assignments "
        "were searched for class balance.",
        "",
        "## Partition counts",
        "",
        "| Partition | Images | Lesions | Approximate image proportion |",
        "|---|---:|---:|---:|",
    ]
    for partition in PARTITION_ORDER:
        lines.append(
            f"| {partition} | {summary['image_counts'][partition]} | "
            f"{summary['lesion_counts'][partition]} | {summary['image_percent'][partition]:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            "| Check | Status | Details |",
            "|---|---|---|",
        ]
    )
    for result in summary["invariants"]:
        lines.append(f"| {result['check']} | {result['status']} | {result['details']} |")
    lines.extend(
        [
            "",
            "All three pairwise lesion intersections are empty; therefore measured lesion-level leakage is zero. "
            "Every diagnosis class occurs in every partition.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    classes, conflicts = validate_metadata(metadata)
    if not conflicts.empty:
        print(conflicts.to_string(index=False))
        raise SystemExit(
            f"STOP: {conflicts['lesion_id'].nunique()} lesion_id values have inconsistent dx labels"
        )

    image_locations, duplicate_files = image_inventory(args.image_dirs)
    metadata = metadata.copy()
    metadata["source_partition"] = metadata["image_id"].map(image_locations)
    fold = assign_folds(metadata)
    metadata["fold"] = fold
    metadata["partition"] = [partition_for_fold(int(value)) for value in fold]
    manifest_columns = [
        "image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization",
        "source_partition", "fold",
    ]
    frames = {
        partition: metadata.loc[metadata["partition"] == partition, manifest_columns].copy()
        for partition in PARTITION_ORDER
    }
    invariants = invariant_results(metadata, frames, image_locations, duplicate_files)
    failed = [result for result in invariants if result["status"] != "PASS"]
    if failed:
        raise RuntimeError("Split invariants failed before persistence: " + json.dumps(failed))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, str] = {}
    for partition, frame in frames.items():
        path = args.output_dir / f"{partition}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        manifest_hashes[path.name] = sha256(path)

    fold_assignments = metadata[
        ["image_id", "lesion_id", "dx", "source_partition", "fold", "partition"]
    ]
    fold_assignments_path = args.output_dir / "fold_assignments.csv"
    fold_assignments.to_csv(fold_assignments_path, index=False, lineterminator="\n")
    distribution = distribution_table(frames, metadata)
    distribution_path = args.output_dir / "class_distribution.csv"
    distribution.to_csv(distribution_path, index=False, lineterminator="\n")

    summary = {
        "status": "PROPOSED JOURNAL SPLIT V1",
        "review_status": "PENDING SENIOR-DEVELOPER REVIEW",
        "dataset": "HAM10000",
        "metadata_sha256": sha256(args.metadata),
        "metadata_rows": int(len(metadata)),
        "unique_image_ids": int(metadata["image_id"].nunique()),
        "unique_lesion_ids": int(metadata["lesion_id"].nunique()),
        "lesion_ids_with_multiple_dx": 0,
        "classes": classes,
        "algorithm": {
            "class": "sklearn.model_selection.StratifiedGroupKFold",
            "n_splits": 10,
            "shuffle": True,
            "random_state": 113,
            "group": "lesion_id",
            "stratification_target": "dx",
            "assignment": {"test": [0], "validation": [1], "train": list(range(2, 10))},
            "selection_note": "Predetermined rule; no seed or fold search was performed.",
        },
        "generator_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "image_inventory": {
            "available_jpg_ids": len(image_locations),
            "metadata_ids_missing_files": len(set(metadata["image_id"]) - set(image_locations)),
            "available_ids_absent_from_metadata": len(set(image_locations) - set(metadata["image_id"])),
            "duplicate_ids_across_image_dirs": len(duplicate_files),
        },
        "image_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "image_percent": {name: float(100 * len(frame) / len(metadata)) for name, frame in frames.items()},
        "lesion_counts": {name: int(frame["lesion_id"].nunique()) for name, frame in frames.items()},
        "fold_image_counts": {
            str(key): int(value) for key, value in metadata["fold"].value_counts().sort_index().items()
        },
        "fold_lesion_counts": {
            str(fold_number): int(metadata.loc[metadata["fold"] == fold_number, "lesion_id"].nunique())
            for fold_number in range(10)
        },
        "manifest_sha256": manifest_hashes,
        "fold_assignments_sha256": sha256(fold_assignments_path),
        "class_distribution_sha256": sha256(distribution_path),
        "invariants": invariants,
        "invariant_totals": {
            "passed": sum(result["status"] == "PASS" for result in invariants),
            "failed": sum(result["status"] == "FAIL" for result in invariants),
        },
    }
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (args.output_dir / "leakage_audit.md").write_text(
        render_audit(summary), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
