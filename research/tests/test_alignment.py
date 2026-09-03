from __future__ import annotations

import numpy as np
import pytest

from research.code.alignment import AlignmentInputError, align_coefficients


def coefficient_data(rows: int = 200, k: int = 5) -> np.ndarray:
    return np.random.default_rng(91).dirichlet(np.arange(1, k + 1), size=rows)


def test_exact_known_permutation_is_recovered() -> None:
    reference = coefficient_data()
    raw_permutation = np.array([2, 0, 4, 1, 3])
    candidate = reference[:, raw_permutation]
    expected_candidate_for_reference = np.argsort(raw_permutation)
    result = align_coefficients(reference, candidate)
    np.testing.assert_array_equal(result.candidate_column_for_reference, expected_candidate_for_reference)
    np.testing.assert_allclose(result.aligned_candidate, reference, atol=1e-12)
    assert np.all(result.cost_method_matrix == "pearson")


def test_noisy_known_permutation_is_recovered() -> None:
    rng = np.random.default_rng(92)
    reference = coefficient_data(rows=500, k=7)
    raw_permutation = np.array([4, 0, 6, 2, 1, 5, 3])
    candidate = np.clip(reference[:, raw_permutation] + rng.normal(0, 1e-4, reference.shape), 0, None)
    candidate /= candidate.sum(axis=1, keepdims=True)
    result = align_coefficients(reference, candidate)
    np.testing.assert_array_equal(result.candidate_column_for_reference, np.argsort(raw_permutation))


def test_numerically_degenerate_column_uses_recorded_fallback() -> None:
    varying = np.random.default_rng(93).dirichlet(np.ones(3), size=200) * 0.8
    reference = np.column_stack([np.full(200, 0.2), varying])
    candidate = reference.copy()
    result = align_coefficients(reference, candidate)
    assert result.reference_std[0] <= 1e-8
    assert result.cost_method_matrix[0, 0] == "fallback_2x_mae"
    assert int(np.sum(result.cost_method_matrix == "fallback_2x_mae")) == 7
    assert result.summary()["fallback_cost_cells"] == 7


def test_mismatched_k_fails_clearly() -> None:
    with pytest.raises(AlignmentInputError, match="identical shape/K"):
        align_coefficients(coefficient_data(k=5), coefficient_data(k=7))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.5, 0.5]),
        np.array([[0.5, 0.5], [np.nan, 0.0]]),
        np.array([[0.5, 0.5], [np.inf, 0.0]]),
    ],
)
def test_malformed_coefficients_fail_clearly(bad: np.ndarray) -> None:
    valid = np.tile(np.array([[0.5, 0.5]]), (2, 1))
    with pytest.raises(AlignmentInputError):
        align_coefficients(valid, bad)

