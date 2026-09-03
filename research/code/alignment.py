"""Validation-only semantic alignment for stability_v1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import DEGENERATE_STD_THRESHOLD


class AlignmentInputError(ValueError):
    """Raised for incompatible or malformed coefficient matrices."""


def validate_coefficients(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise AlignmentInputError(f"{name} must be two-dimensional; shape={matrix.shape}")
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise AlignmentInputError(f"{name} must have at least two rows and two columns")
    if not np.isfinite(matrix).all():
        raise AlignmentInputError(f"{name} contains NaN or Inf")
    if np.any(matrix < -1e-12):
        raise AlignmentInputError(f"{name} contains negative coefficients")
    row_error = np.max(np.abs(matrix.sum(axis=1) - 1.0))
    if row_error > 1e-5:
        raise AlignmentInputError(f"{name} rows do not sum to one; maximum error={row_error}")
    return matrix


@dataclass
class AlignmentResult:
    cost_matrix: np.ndarray
    cost_method_matrix: np.ndarray
    candidate_column_for_reference: np.ndarray
    aligned_candidate: np.ndarray
    reference_std: np.ndarray
    candidate_std: np.ndarray

    def summary(self) -> dict:
        return {
            "degenerate_std_threshold": DEGENERATE_STD_THRESHOLD,
            "fallback": "2 * mean_absolute_error",
            "candidate_column_for_reference": self.candidate_column_for_reference.tolist(),
            "reference_std": self.reference_std.tolist(),
            "candidate_std": self.candidate_std.tolist(),
            "pearson_cost_cells": int(np.sum(self.cost_method_matrix == "pearson")),
            "fallback_cost_cells": int(np.sum(self.cost_method_matrix == "fallback_2x_mae")),
        }


def align_coefficients(reference: np.ndarray, candidate: np.ndarray) -> AlignmentResult:
    """Align candidate columns to reference order from validation activations."""
    reference = validate_coefficients("reference", reference)
    candidate = validate_coefficients("candidate", candidate)
    if reference.shape != candidate.shape:
        raise AlignmentInputError(
            f"Reference and candidate must have identical shape/K; got {reference.shape} and {candidate.shape}"
        )
    k = reference.shape[1]
    reference_std = np.std(reference, axis=0, ddof=0)
    candidate_std = np.std(candidate, axis=0, ddof=0)
    costs = np.empty((k, k), dtype=np.float64)
    methods = np.empty((k, k), dtype=object)
    for i in range(k):
        for j in range(k):
            if (
                reference_std[i] <= DEGENERATE_STD_THRESHOLD
                or candidate_std[j] <= DEGENERATE_STD_THRESHOLD
            ):
                costs[i, j] = 2.0 * np.mean(np.abs(reference[:, i] - candidate[:, j]))
                methods[i, j] = "fallback_2x_mae"
            else:
                correlation = float(np.corrcoef(reference[:, i], candidate[:, j])[0, 1])
                if not np.isfinite(correlation):
                    raise AlignmentInputError(
                        f"Pearson correlation unexpectedly non-finite at reference={i}, candidate={j}"
                    )
                costs[i, j] = 1.0 - correlation
                methods[i, j] = "pearson"
    reference_rows, candidate_columns = linear_sum_assignment(costs)
    if not np.array_equal(reference_rows, np.arange(k)):
        raise RuntimeError("Hungarian solver did not return every reference row in order")
    aligned = candidate[:, candidate_columns]
    return AlignmentResult(
        cost_matrix=costs,
        cost_method_matrix=methods,
        candidate_column_for_reference=candidate_columns,
        aligned_candidate=aligned,
        reference_std=reference_std,
        candidate_std=candidate_std,
    )

