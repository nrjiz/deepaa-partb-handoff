"""Single-run stability_v1 training/export pipeline using frozen manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .constants import (
    BASELINE_COMMIT,
    BASELINE_SOURCE,
    BASELINE_SOURCE_SHA256,
    BATCH_SIZE,
    EPOCHS,
    EXPECTED_CLASSES,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    MANIFEST_SHA256,
    PRIMARY_K,
    PRIMARY_SEEDS,
)
from .data import identity_mapping, load_journal_v1, make_tf_dataset
from .determinism import configure_determinism
from .model_adapter import build_model, provenance, source_sha256
from .output_contract import (
    mark_completed,
    mark_failed,
    require_scoped_output,
    reserve_run_directory,
    write_json,
)
from .stability_metrics import coefficient_weighted_class_distribution, nearest_real_top_n
from .progress import Progress, write_incremental_history


@dataclass
class RunSpec:
    repository: Path
    k: int
    seed: int
    output_dir: Path
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    image_dirs: list[Path]
    allow_completed_rerun: bool = False
    smoke_test: bool = False
    smoke_rows_per_partition: int = 32
    smoke_epochs: int = 1


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status(repository: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_inventory(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(run_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in {".RUNNING", "artifact_inventory.json"}:
            continue
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        )
    write_json(run_dir / "artifact_inventory.json", {"artifacts": artifacts})


def _mean_components(rows: list[list[float]]) -> dict:
    values = np.asarray(rows, dtype=np.float64).mean(axis=0)
    return dict(zip(("total", "reconstruction_sum", "classification_ce", "kl", "archetype_simplex"), values.tolist()))


def _evaluate_losses(baseline, model, cfg, dataset) -> dict:
    rows: list[list[float]] = []
    for images, labels in dataset:
        values = baseline.eval_step(model, images, labels, cfg)
        rows.append([float(value.numpy()) for value in values])
    return _mean_components(rows)


def _collect_outputs(model, dataset) -> dict[str, np.ndarray]:
    coefficients: list[np.ndarray] = []
    latent_means: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    reconstruction_mse: list[np.ndarray] = []
    for images, batch_labels in dataset:
        output = model(images, training=False)
        coefficients.append(output["a"].numpy())
        latent_means.append(output["mu"].numpy())
        labels.append(batch_labels.numpy())
        predictions.append(np.argmax(output["y_logits"].numpy(), axis=1))
        reconstruction_mse.append(
            np.mean((images.numpy() - output["x_rec"].numpy()) ** 2, axis=(1, 2, 3))
        )
    return {
        "coefficients": np.vstack(coefficients),
        "latent_means": np.vstack(latent_means),
        "labels": np.concatenate(labels),
        "predictions": np.concatenate(predictions),
        "reconstruction_mse_per_pixel": np.concatenate(reconstruction_mse),
    }


def _save_partition_outputs(run_dir: Path, name: str, frame: pd.DataFrame, values: dict) -> None:
    np.save(run_dir / f"{name}_coefficients.npy", values["coefficients"], allow_pickle=False)
    np.save(run_dir / f"{name}_latent_means.npy", values["latent_means"], allow_pickle=False)
    mapping = identity_mapping(frame)
    mapping["true_label_index"] = values["labels"]
    mapping["predicted_label_index"] = values["predictions"]
    mapping["predicted_dx"] = [EXPECTED_CLASSES[index] for index in values["predictions"]]
    mapping["reconstruction_mse_per_pixel"] = values["reconstruction_mse_per_pixel"]
    mapping.to_csv(run_dir / f"{name}_mapping.csv", index=False, lineterminator="\n")


def _save_raw_interpretability(run_dir: Path, model, test_frame: pd.DataFrame, test_values: dict) -> None:
    import tensorflow as tf

    z_fixed = model.z_fixed.numpy()
    decoded, _ = model.decode(tf.constant(z_fixed, tf.float32), training=False)
    decoded = decoded.numpy()
    np.save(run_dir / "decoded_archetypes_raw.npy", decoded, allow_pickle=False)
    for index, image in enumerate(decoded, start=1):
        png = tf.io.encode_png(tf.image.convert_image_dtype(image, tf.uint8, saturate=True))
        tf.io.write_file(str(run_dir / f"decoded_archetype_raw_A{index}.png"), png)

    nearest = nearest_real_top_n(
        test_values["latent_means"],
        z_fixed,
        test_frame["image_id"],
        test_frame["dx"],
    )
    pd.DataFrame([row for archetype_rows in nearest for row in archetype_rows]).to_csv(
        run_dir / "test_nearest_real_top10_raw.csv", index=False, lineterminator="\n"
    )
    distributions = coefficient_weighted_class_distribution(
        test_values["coefficients"], test_frame["dx"], EXPECTED_CLASSES
    )
    test_prevalence = (
        test_frame["dx"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0).to_numpy(dtype=float)
        / len(test_frame)
    )
    distribution_rows = []
    for archetype_index in range(distributions.shape[0]):
        for class_index, diagnosis in enumerate(EXPECTED_CLASSES):
            distribution_rows.append(
                {
                    "raw_archetype": archetype_index + 1,
                    "dx": diagnosis,
                    "coefficient_weighted_probability": float(distributions[archetype_index, class_index]),
                    "test_dx_prevalence": float(test_prevalence[class_index]),
                    "enrichment_ratio": (
                        None
                        if test_prevalence[class_index] == 0
                        else float(
                            distributions[archetype_index, class_index]
                            / test_prevalence[class_index]
                        )
                    ),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(
        run_dir / "test_archetype_class_distribution_raw.csv", index=False, lineterminator="\n"
    )


def _checkpoint_reload_check(repository: Path, checkpoint: Path, k: int, trained_model) -> dict:
    import tensorflow as tf

    _, reloaded, _ = build_model(repository, k)
    reloaded(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32), training=False)
    reloaded.load_weights(str(checkpoint))
    maximum_difference = max(
        float(np.max(np.abs(left.numpy() - right.numpy())))
        for left, right in zip(trained_model.weights, reloaded.weights)
    )
    return {
        "loaded": True,
        "weight_tensor_count": len(trained_model.weights),
        "maximum_absolute_weight_difference": maximum_difference,
        "passed": maximum_difference == 0.0,
    }


def execute_run(spec: RunSpec) -> Path:
    """Execute one primary run or one explicitly marked non-scientific smoke run."""
    progress = Progress()
    progress.phase("preflight")
    repository = spec.repository.expanduser().resolve()
    if spec.k not in PRIMARY_K:
        raise ValueError(f"stability_v1 K must be one of {PRIMARY_K}; got {spec.k}")
    if spec.seed not in PRIMARY_SEEDS:
        raise ValueError(f"stability_v1 seed must be one of {PRIMARY_SEEDS}; got {spec.seed}")
    if spec.smoke_test and (spec.smoke_epochs > 1 or spec.smoke_rows_per_partition > BATCH_SIZE):
        raise ValueError("Smoke tests are limited to one epoch and at most one batch per partition")

    source = repository / BASELINE_SOURCE
    before_source_hash = source_sha256(source)
    if before_source_hash != BASELINE_SOURCE_SHA256:
        raise RuntimeError(
            f"Protected conference source hash mismatch before run: expected={BASELINE_SOURCE_SHA256}; actual={before_source_hash}"
        )
    implementation_git_commit = _git_head(repository)
    implementation_git_status = _git_status(repository)
    if implementation_git_status and not spec.smoke_test:
        raise RuntimeError(
            "Primary runs require a clean versioned worktree; current git status:\n"
            + implementation_git_status
        )
    progress.phase("loading_manifests")
    frames = load_journal_v1(
        spec.train_manifest,
        spec.validation_manifest,
        spec.test_manifest,
        spec.image_dirs,
    )
    active_frames = (
        {name: frame.iloc[: spec.smoke_rows_per_partition].copy() for name, frame in frames.items()}
        if spec.smoke_test
        else frames
    )
    scoped_output = require_scoped_output(repository, spec.output_dir, smoke=spec.smoke_test)
    run_dir = reserve_run_directory(
        scoped_output, allow_completed_rerun=spec.allow_completed_rerun
    )
    started = datetime.now(timezone.utc)
    try:
        progress.bind(run_dir)
        progress.phase("initialization_and_dataset_setup")
        determinism = configure_determinism(spec.seed)
        import tensorflow as tf

        baseline, model, cfg = build_model(repository, spec.k)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        manifest_records = {
            name: {
                "path": str(Path(getattr(spec, f"{name}_manifest")).resolve()),
                "sha256": frame.attrs["manifest_sha256"],
                "full_rows": len(frame),
                "used_rows": len(active_frames[name]),
            }
            for name, frame in frames.items()
        }
        configuration = {
            "protocol": "archetype_stability_v1",
            "status": "NON-SCIENTIFIC SMOKE TEST" if spec.smoke_test else "PRIMARY RUN",
            "conference_baseline_commit": BASELINE_COMMIT,
            "implementation_git_commit": implementation_git_commit,
            "implementation_git_status_before_run": implementation_git_status,
            "command_argv": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd().resolve()),
            "model_provenance": provenance(),
            "K": spec.k,
            "latent_dimension": spec.k - 1,
            "seed": spec.seed,
            "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
            "batch_size": BATCH_SIZE,
            "epochs": spec.smoke_epochs if spec.smoke_test else EPOCHS,
            "optimizer": "tf.keras.optimizers.Adam",
            "learning_rate": LEARNING_RATE,
            "early_stopping": False,
            "checkpoint_rule": "final epoch",
            "loss_weights": LOSS_WEIGHTS,
            "raw_unaligned_archetype_numbering": True,
            "manifests": manifest_records,
            "determinism": determinism.to_dict(),
            "process_determinism_environment": {
                "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            },
            "smoke_test": spec.smoke_test,
            "scientific_evidence": False if spec.smoke_test else None,
        }
        write_json(run_dir / "configuration.json", configuration)

        train_dataset = make_tf_dataset(
            active_frames["train"], IMAGE_SIZE, BATCH_SIZE, shuffle=True, seed=spec.seed
        )
        validation_dataset = make_tf_dataset(
            active_frames["validation"], IMAGE_SIZE, BATCH_SIZE, shuffle=False, seed=None
        )
        epochs = spec.smoke_epochs if spec.smoke_test else EPOCHS
        history_rows: list[dict] = []
        epoch_durations: list[float] = []
        steps_per_epoch = (len(active_frames["train"]) + BATCH_SIZE - 1) // BATCH_SIZE
        for epoch in range(1, epochs + 1):
            epoch_started = perf_counter()
            progress.emit("epoch_start", epoch=epoch, epochs=epochs)
            progress.phase("training")
            train_rows: list[list[float]] = []
            for batch, (images, labels) in enumerate(train_dataset, start=1):
                values = baseline.train_step(model, optimizer, images, labels, cfg)
                train_rows.append([float(value.numpy()) for value in values])
                if batch == 1 or batch % 50 == 0 or batch == steps_per_epoch:
                    progress.emit("batch", epoch=epoch, epochs=epochs,
                                  batch=batch, batches=steps_per_epoch)
            train_metrics = _mean_components(train_rows)
            progress.phase("epoch_validation")
            validation_metrics = _evaluate_losses(baseline, model, cfg, validation_dataset)
            history_rows.append(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                }
            )
            progress.phase("epoch_history_export")
            write_incremental_history(run_dir, history_rows)
            epoch_durations.append(perf_counter() - epoch_started)
            progress.emit("epoch_end", epoch=epoch, epochs=epochs,
                          duration_seconds=epoch_durations[-1],
                          train=train_metrics, validation=validation_metrics,
                          remaining_training_loop_seconds=sum(epoch_durations) / epoch * (epochs - epoch),
                          eta_scope="remaining epochs including validation/history; excludes final evaluation/export")

        progress.phase("checkpoint_export_and_reload")
        checkpoint = run_dir / "final.weights.h5"
        model.save_weights(str(checkpoint))
        reload_result = _checkpoint_reload_check(repository, checkpoint, spec.k, model)
        write_json(run_dir / "checkpoint_reload.json", reload_result)
        if not reload_result["passed"]:
            raise RuntimeError("Saved checkpoint did not reload with identical weights")

        split_metrics: dict[str, dict] = {}
        outputs: dict[str, dict] = {}
        for name, frame in active_frames.items():
            progress.phase(f"final_{name}_loss_evaluation")
            ordered_dataset = make_tf_dataset(
                frame, IMAGE_SIZE, BATCH_SIZE, shuffle=False, seed=None
            )
            split_metrics[name] = _evaluate_losses(baseline, model, cfg, ordered_dataset)
            progress.phase(f"final_{name}_output_collection")
            ordered_dataset = make_tf_dataset(
                frame, IMAGE_SIZE, BATCH_SIZE, shuffle=False, seed=None
            )
            outputs[name] = _collect_outputs(model, ordered_dataset)
            split_metrics[name]["accuracy"] = float(
                np.mean(outputs[name]["labels"] == outputs[name]["predictions"])
            )
            split_metrics[name]["reconstruction_mse_per_pixel"] = float(
                outputs[name]["reconstruction_mse_per_pixel"].mean()
            )
        progress.phase("artifact_export")
        write_json(run_dir / "split_metrics.json", split_metrics)
        _save_partition_outputs(run_dir, "validation", active_frames["validation"], outputs["validation"])
        _save_partition_outputs(run_dir, "test", active_frames["test"], outputs["test"])
        _save_raw_interpretability(run_dir, model, active_frames["test"], outputs["test"])

        warnings = determinism.warnings
        write_json(run_dir / "warnings.json", warnings)
        completed = datetime.now(timezone.utc)
        runtime = {
            "started_utc": started.isoformat(),
            "completed_utc": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": importlib.metadata.version("keras"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "h5py": importlib.metadata.version("h5py"),
            "pillow": importlib.metadata.version("Pillow"),
            "devices": [str(device) for device in tf.config.list_physical_devices()],
            "tensorflow_build_info": {
                key: str(value) for key, value in tf.sysconfig.get_build_info().items()
            },
            "inter_op_parallelism_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "intra_op_parallelism_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "determinism": determinism.to_dict(),
        }
        write_json(run_dir / "runtime_environment.json", runtime)
        # Close telemetry before hashing it; no progress-file writes after sealing.
        progress.phase(None)
        _write_artifact_inventory(run_dir)
        after_source_hash = source_sha256(source)
        if after_source_hash != before_source_hash:
            raise RuntimeError("Protected conference source changed during the run")
        mark_completed(
            run_dir,
            {
                "status": "COMPLETED",
                "scientific_evidence": False if spec.smoke_test else None,
                "completed_utc": completed.isoformat(),
                "baseline_source_sha256_before": before_source_hash,
                "baseline_source_sha256_after": after_source_hash,
            },
        )
        print(f"COMPLETED: {run_dir}", flush=True)
        return run_dir
    except Exception as exc:
        mark_failed(
            run_dir,
            {
                "status": "FAILED",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
