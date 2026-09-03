"""Explicit seed and TensorFlow determinism controls."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DeterminismRecord:
    seed: int
    python_hash_seed: str | None
    python_random_seed: int
    numpy_seed: int
    tensorflow_seed: int
    keras_seed: int
    dataset_shuffle_seed: int
    model_initialization_seed: int
    posterior_sampling_seed: int
    deterministic_ops_requested: bool
    deterministic_ops_enabled: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def configure_determinism(seed: int) -> DeterminismRecord:
    """Set all in-process RNGs before model construction and record failures."""
    import tensorflow as tf

    warnings: list[str] = []
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    enabled = False
    try:
        tf.config.experimental.enable_op_determinism()
        enabled = True
    except Exception as exc:  # records environment-specific support failures
        warnings.append(f"enable_op_determinism failed: {type(exc).__name__}: {exc}")
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(seed):
        warnings.append(
            f"PYTHONHASHSEED={hash_seed!r}; expected {seed!r}. Use the CLI wrapper, which re-executes before TensorFlow import."
        )
    return DeterminismRecord(
        seed=seed,
        python_hash_seed=hash_seed,
        python_random_seed=seed,
        numpy_seed=seed,
        tensorflow_seed=seed,
        keras_seed=seed,
        dataset_shuffle_seed=seed,
        model_initialization_seed=seed,
        posterior_sampling_seed=seed,
        deterministic_ops_requested=True,
        deterministic_ops_enabled=enabled,
        warnings=warnings,
    )


def process_environment(seed: int) -> dict[str, str]:
    """Return environment updates that must exist before TensorFlow import."""
    return {
        "PYTHONHASHSEED": str(seed),
        "TF_DETERMINISTIC_OPS": "1",
    }
