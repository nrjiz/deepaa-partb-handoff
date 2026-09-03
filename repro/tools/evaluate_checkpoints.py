#!/usr/bin/env python3
"""Evaluation-only verification of the four committed conference checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.dont_write_bytecode = True


RUNS = {
    3: "20260508_002312",
    5: "20260507_131506",
    7: "20260508_104147",
    9: "20260509_112437",
}

EXPECTED = {
    3: {"mse": 0.008679, "kl": 9.052050, "simplex_penalty": 0.333270, "accuracy": 0.666334,
        "coefficient_variance": 0.022459, "normalized_entropy": 0.897674, "sparsity": 0.102326,
        "average_active_archetypes": 2.825174825174825},
    5: {"mse": 0.006598, "kl": 18.027552, "simplex_penalty": 0.558307, "accuracy": 0.672328,
        "coefficient_variance": 0.010752, "normalized_entropy": 0.913765, "sparsity": 0.086235,
        "average_active_archetypes": 4.193806193806194},
    7: {"mse": 0.004913, "kl": 27.029241, "simplex_penalty": 0.653163, "accuracy": 0.695305,
        "coefficient_variance": 0.006778, "normalized_entropy": 0.912072, "sparsity": 0.087928,
        "average_active_archetypes": 4.578421578421579},
    9: {"mse": 0.004175, "kl": 36.024504, "simplex_penalty": 0.710956, "accuracy": 0.709291,
        "coefficient_variance": 0.004350, "normalized_entropy": 0.916249, "sparsity": 0.083751,
        "average_active_archetypes": 4.373626373626373},
}


def entropy_rows(weights: np.ndarray) -> np.ndarray:
    return -(weights * np.log(weights + 1e-8)).sum(axis=1) / np.log(weights.shape[1])


def make_test_frame(manifest: Path, image_dirs: list[Path]) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    by_name = {directory.name: directory for directory in image_dirs}
    missing_partitions = set(frame["source_partition"]).difference(by_name)
    if missing_partitions:
        raise ValueError(f"No image directory supplied for {sorted(missing_partitions)}")
    frame["filepath"] = [
        str(by_name[source] / f"{image_id}.jpg")
        for source, image_id in zip(frame["source_partition"], frame["image_id"])
    ]
    missing_images = [path for path in frame["filepath"] if not Path(path).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} test images; first: {missing_images[0]}")
    return frame


def committed_metrics(repository: Path, k: int, run: str) -> dict[str, float]:
    tables = repository / "Results_HAM_DAA" / run / "tables"
    table1 = pd.read_csv(tables / "table1_reconstruction_regularization_metrics.csv").iloc[0]
    table6 = pd.read_csv(tables / "table6_coefficient_sparsity_simplex_usage.csv").iloc[0]
    return {
        "mse": float(table1["Recon_MSE_per_pixel_down"]),
        "kl": float(table1["KL_down"]),
        "simplex_penalty": float(table1["Simplex_Penalty_down"]),
        "accuracy": float(table1["Accuracy_optional"]),
        "coefficient_variance": float(table1["Mean_Var_A_down"]),
        "normalized_entropy": float(table6["Mean_Entropy_down"]),
        "sparsity": float(table6["Sparsity_up"]),
        "average_active_archetypes": float(table6["Avg_Active_Archetypes_down"]),
    }


def evaluate_one(baseline, repository: Path, test_frame: pd.DataFrame, k: int, run: str, batch_size: int):
    cfg = baseline.DAAConfig(img_size=128, latent_dim=k - 1, num_classes=7)
    model = baseline.DeepAA(cfg)
    model(tf.zeros((1, 128, 128, 3), dtype=tf.float32), training=False)
    checkpoint = repository / "Results_HAM_DAA" / run / "daa_ham_weights.weights.h5"
    model.load_weights(str(checkpoint))

    dataset = baseline.make_tf_dataset(test_frame, cfg.img_size, batch_size, shuffle=False)
    weights: list[np.ndarray] = []
    reconstruction_mse: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []
    kl_batch_means: list[float] = []
    simplex_batch_means: list[float] = []

    for images, labels in dataset:
        output = model(images, training=False)
        rec = tf.reduce_mean(tf.square(images - output["x_rec"]), axis=[1, 2, 3])
        kl = tf.reduce_mean(baseline.kl_standard_normal(output["mu"], output["logvar"]))
        simplex = tf.reduce_mean(
            tf.reduce_sum(tf.square(model.z_fixed - output["z_pred"]), axis=1)
        )
        weights.append(output["a"].numpy())
        reconstruction_mse.append(rec.numpy())
        true_labels.append(labels.numpy())
        predicted_labels.append(tf.argmax(output["y_logits"], axis=1).numpy())
        kl_batch_means.append(float(kl.numpy()))
        simplex_batch_means.append(float(simplex.numpy()))

    a = np.vstack(weights)
    rec_values = np.concatenate(reconstruction_mse)
    labels = np.concatenate(true_labels)
    predictions = np.concatenate(predicted_labels)
    entropy = entropy_rows(a)
    metrics = {
        "mse": float(rec_values.mean()),
        # Historical table generation averaged already-reduced batch values without batch-size weighting.
        "kl": float(np.mean(kl_batch_means)),
        "simplex_penalty": float(np.mean(simplex_batch_means)),
        "accuracy": float((labels == predictions).mean()),
        "coefficient_variance": float(np.mean(np.var(a, axis=0))),
        "normalized_entropy": float(entropy.mean()),
        "sparsity": float(1.0 - entropy.mean()),
        "average_active_archetypes": float((a >= 0.10).sum(axis=1).mean()),
    }
    details = {
        "K": k,
        "checkpoint": str(checkpoint.relative_to(repository).as_posix()),
        "test_images": int(len(labels)),
        "batch_size": batch_size,
        "batches": len(kl_batch_means),
        "simplex_shape": list(model.z_fixed.shape),
        "minimum_coefficient": float(a.min()),
        "maximum_coefficient_row_sum_error": float(np.max(np.abs(a.sum(axis=1) - 1.0))),
    }
    return metrics, details


def render_markdown(frame: pd.DataFrame, details: list[dict], tolerance: float) -> str:
    lines = [
        "# Checkpoint evaluation verification",
        "",
        "All four committed HDF5 checkpoints were loaded into the unmodified baseline model and evaluated "
        "with deterministic inference (`training=False`) on the reconstructed 1,001-image test split. No "
        "optimization or training was performed.",
        "",
        f"Numeric match tolerance: `{tolerance:g}` absolute. Accuracy is also reported as a fraction of 1,001 images.",
        "",
        "| K | Metric | Published (rounded) | Committed | Reproduced | Absolute difference | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {int(row['K'])} | {row['metric']} | {row['expected_publication_rounded']:.12g} | "
            f"{row['committed_value']:.12g} | {row['checkpoint_reproduced_value']:.12g} | "
            f"{row['absolute_difference']:.3g} | {row['status']} |"
        )
    lines.extend(["", "## Evaluation details", ""])
    for item in details:
        lines.append(
            f"- K={item['K']}: checkpoint `{item['checkpoint']}`, {item['batches']} batches at batch size "
            f"{item['batch_size']}, simplex shape {item['simplex_shape']}, minimum coefficient "
            f"{item['minimum_coefficient']:.3g}, maximum row-sum error {item['maximum_coefficient_row_sum_error']:.3g}."
        )
    lines.extend(
        [
            "",
            "The KL and simplex values intentionally reproduce the historical implementation's mean of batch "
            "means. The simplex term is batch-composition-dependent because `z_pred` uses a softmax across each "
            "batch; changing test order or batch size changes that metric.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--image-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository))
    import daa_ham10000_interpretability_fixed_v2 as baseline

    test_frame = make_test_frame(args.test_manifest, args.image_dirs)
    rows: list[dict] = []
    details: list[dict] = []
    for k, run in RUNS.items():
        metrics, run_details = evaluate_one(
            baseline, repository, test_frame, k, run, args.batch_size
        )
        details.append(run_details)
        saved = committed_metrics(repository, k, run)
        for metric, value in metrics.items():
            difference = abs(saved[metric] - value)
            rows.append(
                {
                    "K": k,
                    "run_directory": run,
                    "metric": metric,
                    "expected_publication_rounded": EXPECTED[k][metric],
                    "committed_value": saved[metric],
                    "checkpoint_reproduced_value": value,
                    "absolute_difference": difference,
                    "tolerance": args.tolerance,
                    "status": "MATCH" if difference <= args.tolerance else "DISCREPANCY",
                }
            )
        print(f"K={k} complete")

    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "checkpoint_metric_verification.csv", index=False, lineterminator="\n")
    (args.output_dir / "checkpoint_metric_verification.md").write_text(
        render_markdown(output, details, args.tolerance), encoding="utf-8", newline="\n"
    )
    (args.output_dir / "checkpoint_evaluation_details.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(output["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
