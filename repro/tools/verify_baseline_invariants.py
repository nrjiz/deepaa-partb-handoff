#!/usr/bin/env python3
"""Small no-training invariant suite for the preserved conference baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
SOURCE_SHA256 = "6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(results: list[dict], name: str, condition: bool, details: str) -> None:
    results.append({"check": name, "status": "PASS" if condition else "FAIL", "details": details})


def import_baseline(source: Path):
    spec = importlib.util.spec_from_file_location("conference_baseline", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    results: list[dict] = []

    source = repository / "daa_ham10000_interpretability_fixed_v2.py"
    actual_source_hash = sha256(source)
    check(
        results, "authoritative_source_hash", actual_source_hash == SOURCE_SHA256,
        f"expected={SOURCE_SHA256}; actual={actual_source_hash}",
    )

    summary = json.loads((args.split_dir / "split_summary.json").read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for name, expected_count in summary["partition_counts"].items():
        manifest = args.split_dir / f"{name}.csv"
        frames[name] = pd.read_csv(manifest)
        actual_hash = sha256(manifest)
        expected_hash = summary["manifest_sha256"][manifest.name]
        check(results, f"{name}_manifest_count", len(frames[name]) == expected_count,
              f"expected={expected_count}; actual={len(frames[name])}")
        check(results, f"{name}_manifest_hash", actual_hash == expected_hash,
              f"expected={expected_hash}; actual={actual_hash}")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(frames[left]["image_id"]) & set(frames[right]["image_id"])
        check(results, f"image_id_disjoint_{left}_{right}", not overlap,
              f"overlap_count={len(overlap)}")

    baseline = import_baseline(source)
    for k, run in RUNS.items():
        tf.keras.backend.clear_session()
        cfg = baseline.DAAConfig(img_size=128, latent_dim=k - 1, num_classes=7)
        model = baseline.DeepAA(cfg)
        synthetic = tf.zeros((2, 128, 128, 3), dtype=tf.float32)
        model(synthetic, training=False)
        checkpoint = repository / "Results_HAM_DAA" / run / "daa_ham_weights.weights.h5"
        load_ok = True
        load_details = str(checkpoint.relative_to(repository).as_posix())
        try:
            model.load_weights(str(checkpoint))
        except Exception as exc:  # pragma: no cover - records an operational failure
            load_ok = False
            load_details += f"; {type(exc).__name__}: {exc}"
        check(results, f"K{k}_checkpoint_load", load_ok, load_details)
        if not load_ok:
            continue
        output = model(synthetic, training=False)
        weights = output["a"].numpy()
        mu = output["mu"].numpy()
        z_fixed = model.z_fixed.numpy()
        check(results, f"K{k}_simplex_shape", z_fixed.shape == (k, k - 1),
              f"actual={list(z_fixed.shape)}; expected={[k, k - 1]}")
        check(results, f"K{k}_coefficients_nonnegative", bool(np.all(weights >= 0.0)),
              f"minimum={weights.min():.9g}")
        row_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
        check(results, f"K{k}_coefficient_rows_sum_to_one", row_error <= 1e-6,
              f"maximum_absolute_error={row_error:.9g}; tolerance=1e-6")
        expected_mu = weights @ z_fixed
        mu_error = float(np.max(np.abs(mu - expected_mu)))
        check(results, f"K{k}_mu_equals_a_times_z_fixed", mu.shape == (2, k - 1) and mu_error <= 1e-6,
              f"shape={list(mu.shape)}; maximum_absolute_error={mu_error:.9g}; tolerance=1e-6")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "source_commit": "29acec505161530f9ea811534b05dd52d8df9899",
        "training_performed": False,
        "total_checks": len(results),
        "passed": sum(item["status"] == "PASS" for item in results),
        "failed": sum(item["status"] == "FAIL" for item in results),
        "checks": results,
    }
    (args.output_dir / "baseline_invariants.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Baseline invariant verification",
        "",
        f"Result: **{output['passed']}/{output['total_checks']} passed**, {output['failed']} failed. No training was performed.",
        "",
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item['check']} | {item['status']} | {item['details']} |" for item in results
    )
    lines.append("")
    (args.output_dir / "baseline_invariants.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    print(f"{output['passed']}/{output['total_checks']} passed; {output['failed']} failed")
    if output["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
