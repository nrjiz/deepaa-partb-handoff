"""Manifest-authoritative HAM10000 input adapter for journal experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EXPECTED_CLASSES, EXPECTED_PARTITION_COUNTS, MANIFEST_SHA256


class ManifestIntegrityError(ValueError):
    """Raised when a frozen manifest or its referenced data fails validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_directory_map(image_dirs: list[Path]) -> dict[str, Path]:
    resolved = [Path(directory).expanduser().resolve() for directory in image_dirs]
    if len({directory.name for directory in resolved}) != len(resolved):
        raise ManifestIntegrityError("Image directory names must be unique")
    missing = [str(directory) for directory in resolved if not directory.is_dir()]
    if missing:
        raise ManifestIntegrityError(f"Image directories do not exist: {missing}")
    return {directory.name: directory for directory in resolved}


def load_manifest(
    manifest_path: Path,
    image_dirs: list[Path],
    expected_hash: str,
    expected_count: int,
) -> pd.DataFrame:
    """Hash-check and load exactly the rows in one frozen manifest, in file order."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    actual_hash = sha256(manifest_path)
    if actual_hash != expected_hash:
        raise ManifestIntegrityError(
            f"Manifest hash mismatch for {manifest_path.name}: expected={expected_hash}; actual={actual_hash}"
        )
    frame = pd.read_csv(manifest_path)
    required = {"image_id", "lesion_id", "dx", "source_partition", "fold"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ManifestIntegrityError(
            f"{manifest_path.name} lacks required columns: {sorted(missing_columns)}"
        )
    if len(frame) != expected_count:
        raise ManifestIntegrityError(
            f"{manifest_path.name} expected {expected_count} rows, found {len(frame)}"
        )
    if frame["image_id"].isna().any() or not frame["image_id"].is_unique:
        raise ManifestIntegrityError(f"{manifest_path.name} contains missing or duplicate image_id")
    if frame["lesion_id"].isna().any():
        raise ManifestIntegrityError(f"{manifest_path.name} contains missing lesion_id")
    observed_classes = set(frame["dx"])
    expected_classes = set(EXPECTED_CLASSES)
    if observed_classes != expected_classes:
        raise ManifestIntegrityError(
            f"{manifest_path.name} diagnosis set differs: "
            f"missing={sorted(expected_classes - observed_classes)}; "
            f"unknown={sorted(observed_classes - expected_classes)}"
        )

    directory_map = _image_directory_map(image_dirs)
    unknown_sources = set(frame["source_partition"]).difference(directory_map)
    if unknown_sources:
        raise ManifestIntegrityError(
            f"No image directory supplied for source partitions: {sorted(unknown_sources)}"
        )
    paths = [
        directory_map[source] / f"{image_id}.jpg"
        for source, image_id in zip(frame["source_partition"], frame["image_id"])
    ]
    missing_files = [str(path) for path in paths if not path.is_file()]
    if missing_files:
        raise ManifestIntegrityError(
            f"{manifest_path.name} references {len(missing_files)} missing JPEGs; first={missing_files[0]}"
        )
    result = frame.copy().reset_index(drop=True)
    result["filepath"] = [str(path) for path in paths]
    class_to_index = {name: index for index, name in enumerate(EXPECTED_CLASSES)}
    result["label"] = result["dx"].map(class_to_index).astype(np.int32)
    result.attrs["manifest_path"] = str(manifest_path)
    result.attrs["manifest_sha256"] = actual_hash
    return result


def load_journal_v1(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    image_dirs: list[Path],
) -> dict[str, pd.DataFrame]:
    """Load all frozen partitions and enforce cross-partition invariants."""
    paths = {
        "train": Path(train_manifest),
        "validation": Path(validation_manifest),
        "test": Path(test_manifest),
    }
    if {path.name for path in paths.values()} != set(MANIFEST_SHA256):
        raise ManifestIntegrityError("Expected train.csv, validation.csv, and test.csv exactly")
    frames = {
        name: load_manifest(
            path,
            image_dirs,
            MANIFEST_SHA256[path.name],
            EXPECTED_PARTITION_COUNTS[name],
        )
        for name, path in paths.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(frames[left]["image_id"]) & set(frames[right]["image_id"]):
            raise ManifestIntegrityError(f"image_id overlap between {left} and {right}")
        if set(frames[left]["lesion_id"]) & set(frames[right]["lesion_id"]):
            raise ManifestIntegrityError(f"lesion_id overlap between {left} and {right}")
    combined = pd.concat(frames.values(), ignore_index=True)
    if len(combined) != sum(EXPECTED_PARTITION_COUNTS.values()) or not combined["image_id"].is_unique:
        raise ManifestIntegrityError("Combined journal_v1 membership is not exactly 10,015 unique images")
    return frames


def make_tf_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int | None,
):
    """Build a deterministic manifest-only pipeline with baseline preprocessing."""
    import tensorflow as tf

    if shuffle and seed is None:
        raise ValueError("A dataset shuffle seed is required")
    paths = frame["filepath"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(frame), 5000),
            seed=int(seed),
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, (image_size, image_size), method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)


def identity_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable identity/label order for coefficient artifacts."""
    return frame[["image_id", "lesion_id", "dx", "source_partition", "fold"]].copy()
