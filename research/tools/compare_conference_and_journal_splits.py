#!/usr/bin/env python3
"""Compare split mechanics and composition without comparing model performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PARTITIONS = ["train", "validation", "test"]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_partitions(directory: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(directory / f"{name}.csv") for name in PARTITIONS}


def pairwise_lesion_overlap(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    sets = {name: set(frame["lesion_id"].astype(str)) for name, frame in frames.items()}
    return {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
        "all_three": len(sets["train"] & sets["validation"] & sets["test"]),
    }


def describe(frames: dict[str, pd.DataFrame]) -> dict:
    total = sum(len(frame) for frame in frames.values())
    return {
        "image_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "image_percent": {name: float(100 * len(frame) / total) for name, frame in frames.items()},
        "lesion_counts": {name: int(frame["lesion_id"].nunique()) for name, frame in frames.items()},
        "pairwise_lesion_overlap": pairwise_lesion_overlap(frames),
    }


def class_comparison(
    historical: dict[str, pd.DataFrame], journal: dict[str, pd.DataFrame]
) -> list[dict]:
    rows: list[dict] = []
    for partition in PARTITIONS:
        for diagnosis in CLASSES:
            historical_count = int((historical[partition]["dx"] == diagnosis).sum())
            journal_count = int((journal[partition]["dx"] == diagnosis).sum())
            historical_percent = 100 * historical_count / len(historical[partition])
            journal_percent = 100 * journal_count / len(journal[partition])
            rows.append(
                {
                    "partition": partition,
                    "dx": diagnosis,
                    "conference_image_count": historical_count,
                    "conference_image_percent": historical_percent,
                    "journal_v1_image_count": journal_count,
                    "journal_v1_image_percent": journal_percent,
                    "journal_minus_conference_percentage_points": journal_percent - historical_percent,
                }
            )
    return rows


def render_markdown(comparison: dict) -> str:
    historical = comparison["historical_conference"]
    journal = comparison["proposed_journal_v1"]
    lines = [
        "# Historical conference split vs proposed journal_v1",
        "",
        "This is a protocol and composition comparison only. No model-performance comparison was performed. "
        "The historical results remain preserved as originally evaluated.",
        "",
        "| Partition | Conference images (%) | journal_v1 images (%) | Conference lesions | journal_v1 lesions |",
        "|---|---:|---:|---:|---:|",
    ]
    for partition in PARTITIONS:
        lines.append(
            f"| {partition} | {historical['image_counts'][partition]} "
            f"({historical['image_percent'][partition]:.4f}%) | {journal['image_counts'][partition]} "
            f"({journal['image_percent'][partition]:.4f}%) | {historical['lesion_counts'][partition]} | "
            f"{journal['lesion_counts'][partition]} |"
        )
    lines.extend(
        [
            "",
            "## Lesion overlap",
            "",
            "| Boundary | Conference shared lesion IDs | journal_v1 shared lesion IDs |",
            "|---|---:|---:|",
        ]
    )
    labels = {
        "train_validation": "train / validation",
        "train_test": "train / test",
        "validation_test": "validation / test",
        "all_three": "all three",
    }
    for key, label in labels.items():
        lines.append(
            f"| {label} | {historical['pairwise_lesion_overlap'][key]} | "
            f"{journal['pairwise_lesion_overlap'][key]} |"
        )
    lines.extend(
        [
            "",
            "## Class composition",
            "",
            "| Partition | dx | Conference count (%) | journal_v1 count (%) | Difference (percentage points) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparison["class_comparison"]:
        lines.append(
            f"| {row['partition']} | {row['dx']} | {row['conference_image_count']} "
            f"({row['conference_image_percent']:.4f}%) | {row['journal_v1_image_count']} "
            f"({row['journal_v1_image_percent']:.4f}%) | "
            f"{row['journal_minus_conference_percentage_points']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "The proposed journal_v1 split retains approximately 80/10/10 image proportions and closely "
            "tracks class proportions while making lesion IDs mutually exclusive. It is intended for future "
            "research evaluation and does not overwrite or relabel the historical conference protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference-dir", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    historical_frames = load_partitions(args.conference_dir)
    journal_frames = load_partitions(args.journal_dir)
    comparison = {
        "scope": "split protocol and composition only; no model performance",
        "historical_conference": describe(historical_frames),
        "proposed_journal_v1": describe(journal_frames),
        "class_comparison": class_comparison(historical_frames, journal_frames),
    }
    (args.journal_dir / "conference_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (args.journal_dir / "conference_comparison.md").write_text(
        render_markdown(comparison), encoding="utf-8", newline="\n"
    )
    print(json.dumps(comparison["historical_conference"], indent=2))
    print(json.dumps(comparison["proposed_journal_v1"], indent=2))


if __name__ == "__main__":
    main()
