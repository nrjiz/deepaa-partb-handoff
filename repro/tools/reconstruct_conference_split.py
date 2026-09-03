#!/usr/bin/env python3
"""Reconstruct and audit the baseline image-level HAM10000 split.

This is an evaluation-only companion. It deliberately mirrors
``stratified_split`` in the authoritative conference source without importing
TensorFlow or changing the historical implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_image_index(image_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in image_dirs:
        for image_path in directory.glob("*.jpg"):
            index[image_path.stem] = image_path
    return index


def load_dataframe(metadata_path: Path, image_dirs: list[Path]) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    image_index = build_image_index(image_dirs)
    paths: list[Path] = []
    keep: list[bool] = []
    for image_id in df["image_id"].astype(str):
        path = image_index.get(image_id)
        keep.append(path is not None)
        if path is not None:
            paths.append(path)

    df = df.loc[np.asarray(keep)].copy()
    df["filepath"] = [str(path) for path in paths]
    df["source_partition"] = [path.parent.name for path in paths]
    df["label_str"] = df["dx"].astype(str)
    classes = sorted(df["label_str"].unique().tolist())
    if classes != CLASSES:
        raise ValueError(f"Unexpected diagnosis classes: {classes}")
    df["label"] = df["label_str"].map({name: i for i, name in enumerate(classes)}).astype(int)
    return df


def historical_split(
    df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exact copy of the indexing and RNG behavior in the baseline source."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    validation_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(df["label"].unique()):
        indices = df.index[df["label"] == label].to_numpy().copy()
        rng.shuffle(indices)
        count = len(indices)
        test_count = int(round(count * test_fraction))
        validation_count = int(round(count * val_fraction))
        test_idx.extend(indices[:test_count].tolist())
        validation_idx.extend(indices[test_count : test_count + validation_count].tolist())
        train_idx.extend(indices[test_count + validation_count :].tolist())
    return df.loc[train_idx], df.loc[validation_idx], df.loc[test_idx]


def manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["image_id", "lesion_id", "dx", "label", "source_partition"]].copy()
    result.insert(0, "metadata_row_index", frame.index.astype(int))
    return result


def count_by_class(frame: pd.DataFrame) -> dict[str, int]:
    return {name: int((frame["dx"] == name).sum()) for name in CLASSES}


def leakage_details(partitions: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    lesion_sets = {name: set(frame["lesion_id"].astype(str)) for name, frame in partitions.items()}
    pair_names = [("train", "validation"), ("train", "test"), ("validation", "test")]
    pairwise: dict[str, dict] = {}
    for left, right in pair_names:
        overlap = lesion_sets[left] & lesion_sets[right]
        pairwise[f"{left}_{right}"] = {
            "overlapping_lesion_ids": len(overlap),
            f"{left}_affected_images": int(partitions[left]["lesion_id"].astype(str).isin(overlap).sum()),
            f"{left}_affected_percent": float(
                100 * partitions[left]["lesion_id"].astype(str).isin(overlap).mean()
            ),
            f"{right}_affected_images": int(partitions[right]["lesion_id"].astype(str).isin(overlap).sum()),
            f"{right}_affected_percent": float(
                100 * partitions[right]["lesion_id"].astype(str).isin(overlap).mean()
            ),
        }

    all_three = lesion_sets["train"] & lesion_sets["validation"] & lesion_sets["test"]
    cross_partition = (
        (lesion_sets["train"] & lesion_sets["validation"])
        | (lesion_sets["train"] & lesion_sets["test"])
        | (lesion_sets["validation"] & lesion_sets["test"])
    )
    combined = pd.concat(
        [frame.assign(split=name) for name, frame in partitions.items()], ignore_index=True
    )
    rows: list[dict] = []
    for lesion_id in sorted(cross_partition):
        lesion_rows = combined[combined["lesion_id"].astype(str) == lesion_id]
        diagnoses = sorted(lesion_rows["dx"].astype(str).unique())
        row = {"lesion_id": lesion_id, "dx": ";".join(diagnoses)}
        for split_name in partitions:
            count = int((lesion_rows["split"] == split_name).sum())
            row[f"{split_name}_images"] = count
            row[f"in_{split_name}"] = count > 0
        row["total_images"] = int(len(lesion_rows))
        row["in_all_three"] = lesion_id in all_three
        rows.append(row)
    details = pd.DataFrame(rows)

    affected_images: dict[str, dict] = {}
    for split_name, frame in partitions.items():
        affected = int(frame["lesion_id"].astype(str).isin(cross_partition).sum())
        affected_images[split_name] = {
            "count": affected,
            "percent": float(100 * affected / len(frame)),
        }

    lesion_dx = details.drop_duplicates("lesion_id").groupby("dx").size() if len(details) else pd.Series(dtype=int)
    report = {
        "pairwise": pairwise,
        "lesions_in_all_three": len(all_three),
        "affected_unique_lesion_ids": len(cross_partition),
        "affected_images_total": int(sum(value["count"] for value in affected_images.values())),
        "affected_images_by_partition": affected_images,
        "overlapping_lesions_by_dx": {str(k): int(v) for k, v in lesion_dx.items()},
    }
    return details, report


def render_leakage_report(report: dict, details_hash: str) -> str:
    lines = [
        "# Historical lesion-level overlap audit",
        "",
        "This measures leakage caused by the conference baseline's diagnosis-stratified, image-level split. "
        "It does not alter the split and does not by itself invalidate the historical results.",
        "",
        "## Pairwise overlap",
        "",
        "| Partitions | Shared lesion IDs | Left affected images (%) | Right affected images (%) |",
        "|---|---:|---:|---:|",
    ]
    for key, value in report["pairwise"].items():
        left, right = key.split("_", 1)
        lines.append(
            f"| {left} / {right} | {value['overlapping_lesion_ids']} | "
            f"{value[f'{left}_affected_images']} ({value[f'{left}_affected_percent']:.4f}%) | "
            f"{value[f'{right}_affected_images']} ({value[f'{right}_affected_percent']:.4f}%) |"
        )
    lines.extend(
        [
            "",
            f"Lesions appearing in all three partitions: **{report['lesions_in_all_three']}**.",
            f"Unique lesion IDs crossing at least one boundary: **{report['affected_unique_lesion_ids']}**.",
            f"Images belonging to those lesions: **{report['affected_images_total']}**.",
            "",
            "## Any-boundary affected images",
            "",
            "| Partition | Affected images | Percent of partition |",
            "|---|---:|---:|",
        ]
    )
    for name, value in report["affected_images_by_partition"].items():
        lines.append(f"| {name} | {value['count']} | {value['percent']:.4f}% |")
    lines.extend(
        [
            "",
            "## Overlapping lesions by diagnosis",
            "",
            "| Diagnosis | Unique overlapping lesion IDs |",
            "|---|---:|",
        ]
    )
    for diagnosis, count in report["overlapping_lesions_by_dx"].items():
        lines.append(f"| {diagnosis} | {count} |")
    lines.extend(
        [
            "",
            f"Machine-readable detail: `leakage_details.csv` (SHA-256 `{details_hash}`).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    args = parser.parse_args()

    dataframe = load_dataframe(args.metadata, args.image_dirs)
    split_frames = historical_split(
        dataframe, args.validation_fraction, args.test_fraction, args.seed
    )
    partitions = dict(zip(("train", "validation", "test"), split_frames))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_hashes: dict[str, str] = {}
    for name, frame in partitions.items():
        destination = args.output_dir / f"{name}.csv"
        manifest_frame(frame).to_csv(destination, index=False, lineterminator="\n")
        manifest_hashes[destination.name] = sha256(destination)

    details, leakage = leakage_details(partitions)
    details_path = args.output_dir / "leakage_details.csv"
    details.to_csv(details_path, index=False, lineterminator="\n")
    details_hash = sha256(details_path)
    (args.output_dir / "leakage_report.md").write_text(
        render_leakage_report(leakage, details_hash), encoding="utf-8", newline="\n"
    )

    summary = {
        "protocol": "conference_image_level",
        "source_commit": "29acec505161530f9ea811534b05dd52d8df9899",
        "algorithm": (
            "One numpy.random.default_rng(seed); for each sorted integer diagnosis label, shuffle metadata "
            "indices, assign round(n*test_fraction) to test, then round(n*validation_fraction) to validation, "
            "and the remainder to train. No lesion_id grouping."
        ),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "metadata_sha256": sha256(args.metadata),
        "available_metadata_rows": int(len(dataframe)),
        "partition_counts": {name: int(len(frame)) for name, frame in partitions.items()},
        "class_counts": {name: count_by_class(frame) for name, frame in partitions.items()},
        "manifest_sha256": manifest_hashes,
        "leakage_details_sha256": details_hash,
        "leakage": leakage,
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
