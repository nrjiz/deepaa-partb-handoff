from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.code.constants import EXPECTED_PARTITION_COUNTS, MANIFEST_SHA256
from research.code.data import ManifestIntegrityError, load_journal_v1, load_manifest, make_tf_dataset


REPOSITORY = Path(__file__).resolve().parents[2]
SPLIT = REPOSITORY / "research" / "splits" / "journal_v1"
ARCHIVE = Path.home() / "Downloads" / "archive"
IMAGE_DIRS = [ARCHIVE / "HAM10000_images_part_1", ARCHIVE / "HAM10000_images_part_2"]


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    if not all(path.is_dir() for path in IMAGE_DIRS):
        pytest.skip("Local HAM10000 image directories are unavailable")
    return load_journal_v1(
        SPLIT / "train.csv", SPLIT / "validation.csv", SPLIT / "test.csv", IMAGE_DIRS
    )


def test_frozen_membership_hashes_counts_and_order(frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        raw = pd.read_csv(SPLIT / f"{name}.csv")
        assert len(frame) == EXPECTED_PARTITION_COUNTS[name]
        assert frame.attrs["manifest_sha256"] == MANIFEST_SHA256[f"{name}.csv"]
        assert frame["image_id"].tolist() == raw["image_id"].tolist()
        assert frame["lesion_id"].tolist() == raw["lesion_id"].tolist()
        assert frame["dx"].tolist() == raw["dx"].tolist()


def test_no_partition_membership_overlap(frames: dict[str, pd.DataFrame]) -> None:
    assert sum(len(frame) for frame in frames.values()) == 10015
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(frames[left]["image_id"]).isdisjoint(frames[right]["image_id"])
        assert set(frames[left]["lesion_id"]).isdisjoint(frames[right]["lesion_id"])


def test_validation_and_test_pipeline_order_is_repeatable(frames: dict[str, pd.DataFrame]) -> None:
    for name in ("validation", "test"):
        subset = frames[name].iloc[:32]
        first = next(iter(make_tf_dataset(subset, 128, 32, shuffle=False, seed=None)))
        second = next(iter(make_tf_dataset(subset, 128, 32, shuffle=False, seed=None)))
        np.testing.assert_array_equal(first[1].numpy(), subset["label"].to_numpy())
        np.testing.assert_array_equal(first[1].numpy(), second[1].numpy())
        np.testing.assert_allclose(first[0].numpy(), second[0].numpy(), atol=0)


def test_hash_mismatch_fails_before_loading(tmp_path: Path) -> None:
    tampered = tmp_path / "train.csv"
    tampered.write_bytes((SPLIT / "train.csv").read_bytes() + b"\n")
    with pytest.raises(ManifestIntegrityError, match="hash mismatch"):
        load_manifest(tampered, IMAGE_DIRS, MANIFEST_SHA256["train.csv"], 8011)

