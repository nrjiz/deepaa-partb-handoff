"""Frozen scientific and integrity constants for stability_v1."""

from __future__ import annotations


BASELINE_COMMIT = "29acec505161530f9ea811534b05dd52d8df9899"
BASELINE_SOURCE = "daa_ham10000_interpretability_fixed_v2.py"
BASELINE_SOURCE_SHA256 = "6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d"

MANIFEST_SHA256 = {
    "train.csv": "4c5a8912d4235f4e91ebbe374113e9db6a9cc919b550b4020b4ce670ecd3acdf",
    "validation.csv": "ac7077667504fdf117d198d87fc01b0922941de37816b3d65947df917fa21deb",
    "test.csv": "7caad8c2d7d92510936d391a8851344d3aa15c68a66ca981c6cd382e2ddb13d7",
}
EXPECTED_PARTITION_COUNTS = {"train": 8011, "validation": 1002, "test": 1002}
EXPECTED_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

PRIMARY_K = (5, 7)
PRIMARY_SEEDS = (113, 211, 307, 401, 503)
REFERENCE_SEED = 113

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
LOSS_WEIGHTS = {
    "reconstruction": 1.0,
    "classification": 1.0,
    "kl": 0.01,
    "archetype_simplex": 1.0,
    "coefficient_entropy": 0.0,
}

DEGENERATE_STD_THRESHOLD = 1e-8
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 2026
NEAREST_REAL_TOP_N = 10
SSIM_DATA_RANGE = 1.0

