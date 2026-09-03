#!/usr/bin/env python3
"""Independently recompute metrics available in committed result CSV files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


RUNS = {
    3: "20260508_002312",
    5: "20260507_131506",
    7: "20260508_104147",
    9: "20260509_112437",
}

EXPECTED = {
    3: {"mse": 0.008679, "kl": 9.052050, "simplex_penalty": 0.333270, "accuracy": 0.666334,
        "coefficient_variance": 0.022459, "normalized_entropy": 0.897674, "sparsity": 0.102326},
    5: {"mse": 0.006598, "kl": 18.027552, "simplex_penalty": 0.558307, "accuracy": 0.672328,
        "coefficient_variance": 0.010752, "normalized_entropy": 0.913765, "sparsity": 0.086235},
    7: {"mse": 0.004913, "kl": 27.029241, "simplex_penalty": 0.653163, "accuracy": 0.695305,
        "coefficient_variance": 0.006778, "normalized_entropy": 0.912072, "sparsity": 0.087928},
    9: {"mse": 0.004175, "kl": 36.024504, "simplex_penalty": 0.710956, "accuracy": 0.709291,
        "coefficient_variance": 0.004350, "normalized_entropy": 0.916249, "sparsity": 0.083751},
}


def entropy_rows(weights: np.ndarray) -> np.ndarray:
    return -(weights * np.log(weights + 1e-8)).sum(axis=1) / np.log(weights.shape[1])


def add_row(
    rows: list[dict], k: int, run: str, metric: str, component: str,
    expected: float | None, committed: float | None, recomputed: float | None,
    tolerance: float | None, status: str, evidence: str,
) -> None:
    rows.append(
        {
            "K": k,
            "run_directory": run,
            "metric": metric,
            "component": component,
            "expected_publication_rounded": expected,
            "committed_value": committed,
            "recomputed_value": recomputed,
            "absolute_difference_expected_vs_committed": (
                abs(expected - committed) if expected is not None and committed is not None else np.nan
            ),
            "absolute_difference_committed_vs_recomputed": (
                abs(committed - recomputed) if committed is not None and recomputed is not None else np.nan
            ),
            "tolerance": tolerance,
            "status": status,
            "evidence": evidence,
        }
    )


def parse_usage(value: str) -> dict[str, float]:
    usage: dict[str, float] = {}
    for part in value.split(";"):
        name, percent = part.strip().split("=")
        usage[name] = float(percent.rstrip("%"))
    return usage


def verify_run(results_root: Path, k: int, run: str, rows: list[dict]) -> None:
    tables = results_root / run / "tables"
    per_image = pd.read_csv(tables / "per_image_archetype_interpretation.csv")
    table1 = pd.read_csv(tables / "table1_reconstruction_regularization_metrics.csv").iloc[0]
    table3 = pd.read_csv(tables / "table3_class_distribution_per_archetype_percent.csv")
    table6 = pd.read_csv(tables / "table6_coefficient_sparsity_simplex_usage.csv").iloc[0]
    a_cols = [f"A{i}_weight" for i in range(1, k + 1)]
    weights = per_image[a_cols].to_numpy(dtype=np.float64)
    tolerance = 1e-7

    committed = {
        "mse": float(table1["Recon_MSE_per_pixel_down"]),
        "kl": float(table1["KL_down"]),
        "simplex_penalty": float(table1["Simplex_Penalty_down"]),
        "accuracy": float(table1["Accuracy_optional"]),
        "coefficient_variance": float(table1["Mean_Var_A_down"]),
        "normalized_entropy": float(table6["Mean_Entropy_down"]),
        "sparsity": float(table6["Sparsity_up"]),
        "average_active_archetypes": float(table6["Avg_Active_Archetypes_down"]),
    }
    recomputed = {
        "mse": float(per_image["recon_mse_per_pixel"].mean()),
        "accuracy": float((per_image["true_label_idx"] == per_image["pred_label_idx"]).mean()),
        "coefficient_variance": float(np.mean(np.var(weights, axis=0))),
        "normalized_entropy": float(entropy_rows(weights).mean()),
        "sparsity": float(1.0 - entropy_rows(weights).mean()),
        "average_active_archetypes": float((weights >= 0.10).sum(axis=1).mean()),
    }

    for metric in ("mse", "accuracy", "coefficient_variance", "normalized_entropy", "sparsity", "average_active_archetypes"):
        difference = abs(committed[metric] - recomputed[metric])
        add_row(
            rows, k, run, metric, "overall", EXPECTED[k].get(metric), committed[metric],
            recomputed[metric], tolerance, "MATCH" if difference <= tolerance else "DISCREPANCY",
            "Recomputed from per_image_archetype_interpretation.csv",
        )

    for metric, column in (("kl", "KL_down"), ("simplex_penalty", "Simplex_Penalty_down")):
        add_row(
            rows, k, run, metric, "overall", EXPECTED[k][metric], float(table1[column]), None,
            None, "NOT_RECOMPUTABLE_FROM_ARTIFACTS",
            "Saved per-image CSV omits log-variance and batch-derived z_pred required for this metric",
        )

    committed_usage = parse_usage(str(table6["Usage_Per_Archetype"]))
    for archetype in [f"A{i}" for i in range(1, k + 1)]:
        actual = float(100 * (per_image["dominant_archetype"] == archetype).mean())
        saved = committed_usage[archetype]
        # The historical string intentionally rounds usage to one decimal place.
        difference = abs(saved - actual)
        add_row(
            rows, k, run, "archetype_usage_percent", archetype, None, saved, actual, 0.0500001,
            "MATCH_ROUNDED" if difference <= 0.0500001 else "DISCREPANCY",
            "Committed table stores one decimal place; recomputed from dominant_archetype",
        )

    classes = [column for column in table3.columns if column != "Archetype"]
    computed_table3 = (
        pd.crosstab(per_image["dominant_archetype"], per_image["true_label"], normalize="index") * 100
    ).reindex([f"A{i}" for i in range(1, k + 1)]).fillna(0)
    saved_table3 = table3.set_index("Archetype")
    for archetype in [f"A{i}" for i in range(1, k + 1)]:
        for diagnosis in classes:
            saved = float(saved_table3.loc[archetype, diagnosis])
            actual = float(computed_table3.loc[archetype, diagnosis])
            difference = abs(saved - actual)
            add_row(
                rows, k, run, "class_distribution_percent", f"{archetype}:{diagnosis}", None,
                saved, actual, 1e-10, "MATCH" if difference <= 1e-10 else "DISCREPANCY",
                "Recomputed crosstab from dominant_archetype and true_label",
            )


def markdown_report(frame: pd.DataFrame) -> str:
    overall = frame[(frame["component"] == "overall")]
    lines = [
        "# Artifact-only metric verification",
        "",
        "Metrics were recomputed only from committed CSV artifacts; historical result files were not changed. "
        "The tolerance for numeric CSV recomputation is `1e-7`. Published expected values are rounded and are "
        "shown for provenance, not judged with that tolerance.",
        "",
        "| K | Metric | Published (rounded) | Committed | Recomputed | Absolute difference | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in overall.iterrows():
        def show(value: object) -> str:
            return "N/A" if pd.isna(value) else f"{float(value):.12g}"
        lines.append(
            f"| {int(row['K'])} | {row['metric']} | {show(row['expected_publication_rounded'])} | "
            f"{show(row['committed_value'])} | {show(row['recomputed_value'])} | "
            f"{show(row['absolute_difference_committed_vs_recomputed'])} | {row['status']} |"
        )
    status_counts = frame["status"].value_counts().to_dict()
    lines.extend(
        [
            "",
            "Archetype usage was checked against the historical one-decimal strings with a 0.05 percentage-point "
            "rounding allowance. Every cell of each class-by-archetype distribution table was also recomputed; "
            "those detailed comparisons are in the CSV.",
            "",
            "KL cannot be independently recovered from CSVs because per-image posterior `logvar` was not saved. "
            "The simplex/archetype penalty cannot be recovered because batch-derived `z_pred` was not saved. "
            "Both require checkpoint evaluation.",
            "",
            "Status counts: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())) + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for k, run in RUNS.items():
        verify_run(args.results_root, k, run, rows)
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "artifact_metric_verification.csv", index=False, lineterminator="\n")
    (args.output_dir / "artifact_metric_verification.md").write_text(
        markdown_report(frame), encoding="utf-8", newline="\n"
    )
    print(frame["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
