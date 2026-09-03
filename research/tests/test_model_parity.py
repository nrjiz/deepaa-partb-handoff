from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from research.code.constants import BASELINE_SOURCE_SHA256, LOSS_WEIGHTS
from research.code.model_adapter import build_model, load_baseline_module, source_sha256


REPOSITORY = Path(__file__).resolve().parents[2]


def test_protected_source_hash() -> None:
    assert source_sha256(REPOSITORY / "daa_ham10000_interpretability_fixed_v2.py") == BASELINE_SOURCE_SHA256


@pytest.mark.parametrize("k", [5, 7])
def test_model_simplex_variational_and_loss_invariants(k: int) -> None:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(113)
    baseline, model, cfg = build_model(REPOSITORY, k)
    images = tf.zeros((2, 128, 128, 3), dtype=tf.float32)
    inference = model(images, training=False)
    sampled = model(images, training=True)
    coefficients = inference["a"].numpy()
    assert model.k == k
    assert cfg.latent_dim == k - 1
    assert model.z_fixed.shape == (k, k - 1)
    assert np.all(coefficients >= 0)
    np.testing.assert_allclose(coefficients.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        inference["mu"].numpy(), coefficients @ model.z_fixed.numpy(), atol=1e-6
    )
    assert sampled["t"].shape == (2, k - 1)
    assert cfg.recon_w == LOSS_WEIGHTS["reconstruction"]
    assert cfg.class_w == LOSS_WEIGHTS["classification"]
    assert cfg.kl_w == LOSS_WEIGHTS["kl"]
    assert cfg.at_w == LOSS_WEIGHTS["archetype_simplex"]
    values = baseline.eval_step(model, images, tf.zeros(2, dtype=tf.int32), cfg)
    assert len(values) == 5
    assert all(np.isfinite(float(value.numpy())) for value in values)


def test_journal_adapter_and_direct_baseline_are_identical() -> None:
    baseline = load_baseline_module(REPOSITORY)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(307)
    _, adapted, cfg = build_model(REPOSITORY, 5)
    inputs = tf.zeros((1, 128, 128, 3), dtype=tf.float32)
    adapted_output = adapted(inputs, training=False)
    adapted_weights = [weight.numpy().copy() for weight in adapted.weights]

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(307)
    direct_cfg = baseline.DAAConfig(
        img_size=128, latent_dim=4, num_classes=7,
        recon_w=1.0, class_w=1.0, kl_w=0.01, at_w=1.0,
    )
    direct = baseline.DeepAA(direct_cfg)
    direct_output = direct(inputs, training=False)
    for left, right in zip(adapted_weights, direct.weights):
        np.testing.assert_array_equal(left, right.numpy())
    for key in adapted_output:
        np.testing.assert_array_equal(adapted_output[key].numpy(), direct_output[key].numpy())
    assert cfg == direct_cfg

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(401)
    _, different_seed_model, _ = build_model(REPOSITORY, 5)
    different_seed_model(inputs, training=False)
    assert any(
        not np.array_equal(reference, changed.numpy())
        for reference, changed in zip(adapted_weights, different_seed_model.weights)
    )


def test_entropy_is_not_a_training_objective() -> None:
    baseline = load_baseline_module(REPOSITORY)
    training_source = inspect.getsource(baseline.train_step).lower()
    assert "entropy_rows" not in training_source
    assert "gini_like_sparsity" not in training_source
    assert re.search(r"\ba_entropy\b", training_source) is None
    assert re.search(r"\bentropy_w\b", training_source) is None
    assert not hasattr(baseline.DAAConfig(), "entropy_w")
    assert LOSS_WEIGHTS["coefficient_entropy"] == 0.0
