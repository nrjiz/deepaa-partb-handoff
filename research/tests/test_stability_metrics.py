from __future__ import annotations

import numpy as np
import pandas as pd

from research.code.stability_metrics import (
    aggregate_by_lesion,
    coefficient_stability,
    compare_nearest_real,
    lesion_bootstrap_coefficient_stability,
    nearest_real_top_n,
    semantic_stability,
    summarize_seed_comparisons,
    visual_stability,
)


def mapping(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [f"image_{index}" for index in range(rows)],
            "lesion_id": [f"lesion_{index // 2}" for index in range(rows)],
            "dx": ["nv" if index // 2 % 2 == 0 else "mel" for index in range(rows)],
        }
    )


def test_lesion_aggregation_is_arithmetic_mean() -> None:
    values = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9], [0.3, 0.7]])
    aggregated, identities = aggregate_by_lesion(values, mapping(4))
    np.testing.assert_allclose(aggregated, [[0.7, 0.3], [0.2, 0.8]])
    assert identities["image_count"].tolist() == [2, 2]


def test_identical_aligned_coefficients_have_perfect_stability() -> None:
    coefficients = np.random.default_rng(4).dirichlet(np.ones(5), size=40)
    result = coefficient_stability(coefficients, mapping(40), coefficients.copy(), mapping(40))
    assert result["coefficient_vector_l1_mean"] == 0.0
    assert result["dominant_archetype_agreement"] == 1.0
    assert all(item["mae"] == 0.0 for item in result["per_archetype"])
    np.testing.assert_allclose(
        [item["pearson_correlation"] for item in result["per_archetype"]], 1.0
    )


def test_identical_semantics_have_zero_js_divergence() -> None:
    coefficients = np.random.default_rng(5).dirichlet(np.ones(5), size=40)
    labels = mapping(40)["dx"]
    result = semantic_stability(coefficients, labels, coefficients.copy(), labels)
    np.testing.assert_allclose(result["jensen_shannon_divergence_base2"], 0.0, atol=1e-15)


def test_identical_nearest_sets_have_unit_jaccard() -> None:
    rng = np.random.default_rng(6)
    latent = rng.normal(size=(30, 2))
    vertices = np.array([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]])
    ids = [f"image_{i}" for i in range(30)]
    labels = ["nv" if i % 2 else "mel" for i in range(30)]
    nearest = nearest_real_top_n(latent, vertices, ids, labels)
    compared = compare_nearest_real(nearest, nearest, [0, 1, 2])
    assert all(item["image_id_jaccard"] == 1.0 for item in compared)
    np.testing.assert_allclose(
        [item["diagnosis_jensen_shannon_divergence_base2"] for item in compared], 0.0
    )


def test_rgb_ssim_and_pixel_mse_for_identical_images() -> None:
    images = np.random.default_rng(7).random((3, 128, 128, 3), dtype=np.float32)
    result = visual_stability(images, images.copy(), [0, 1, 2], data_range=1.0)
    np.testing.assert_allclose([row["rgb_ssim"] for row in result], 1.0, atol=1e-6)
    np.testing.assert_allclose([row["pixel_mse"] for row in result], 0.0)
    assert all(row["data_range"] == 1.0 for row in result)


def test_lesion_bootstrap_is_seeded_and_uses_requested_resamples() -> None:
    reference = np.random.default_rng(8).dirichlet(np.ones(5), size=25)
    candidate = np.roll(reference, 1, axis=0)
    first = lesion_bootstrap_coefficient_stability(reference, candidate, resamples=25, seed=2026)
    second = lesion_bootstrap_coefficient_stability(reference, candidate, resamples=25, seed=2026)
    assert first == second
    assert first["unit"] == "lesion_id"
    assert first["resamples"] == 25


def test_seed_comparison_summary_uses_sample_standard_deviation() -> None:
    result = summarize_seed_comparisons([1.0, 2.0, 3.0, 4.0])
    assert result["raw"] == [1.0, 2.0, 3.0, 4.0]
    assert result["sample_standard_deviation"] == np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    assert result["inferential_interval_across_seed_comparisons"] is None
