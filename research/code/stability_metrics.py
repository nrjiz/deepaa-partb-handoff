"""Held-out stability metrics, independent from model training."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .constants import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEGENERATE_STD_THRESHOLD,
    EXPECTED_CLASSES,
    NEAREST_REAL_TOP_N,
    SSIM_DATA_RANGE,
)


class StabilityInputError(ValueError):
    """Raised when held-out artifacts cannot be paired safely."""


def _permutation(values: Sequence[int], k: int) -> np.ndarray:
    permutation = np.asarray(values, dtype=int)
    if permutation.shape != (k,) or not np.array_equal(np.sort(permutation), np.arange(k)):
        raise StabilityInputError(f"Permutation must contain every column 0..{k - 1} exactly once")
    return permutation


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise StabilityInputError(f"{name} must be a finite two-dimensional matrix")
    return result


def aggregate_by_lesion(
    coefficients: np.ndarray, mapping: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Arithmetic-mean image coefficients within lesion, preserving first-seen order."""
    coefficients = _matrix(coefficients, "coefficients")
    if len(coefficients) != len(mapping):
        raise StabilityInputError("Coefficient rows and mapping rows differ")
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(mapping.columns):
        raise StabilityInputError(f"Mapping lacks {sorted(required.difference(mapping.columns))}")
    rows: list[np.ndarray] = []
    identities: list[dict] = []
    work = mapping.reset_index(drop=True)
    for lesion_id in pd.unique(work["lesion_id"]):
        indices = np.flatnonzero(work["lesion_id"].to_numpy() == lesion_id)
        labels = work.iloc[indices]["dx"].unique()
        if len(labels) != 1:
            raise StabilityInputError(f"Lesion {lesion_id} has inconsistent dx labels")
        rows.append(coefficients[indices].mean(axis=0))
        identities.append(
            {"lesion_id": lesion_id, "dx": labels[0], "image_count": int(len(indices))}
        )
    return np.vstack(rows), pd.DataFrame(identities)


def paired_lesion_coefficients(
    reference: np.ndarray,
    reference_mapping: pd.DataFrame,
    aligned_candidate: np.ndarray,
    candidate_mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ref_values, ref_ids = aggregate_by_lesion(reference, reference_mapping)
    cand_values, cand_ids = aggregate_by_lesion(aligned_candidate, candidate_mapping)
    if set(ref_ids["lesion_id"]) != set(cand_ids["lesion_id"]):
        raise StabilityInputError("Reference and candidate lesion sets differ")
    lookup = {lesion: index for index, lesion in enumerate(cand_ids["lesion_id"])}
    order = np.asarray([lookup[lesion] for lesion in ref_ids["lesion_id"]], dtype=int)
    cand_ids = cand_ids.iloc[order].reset_index(drop=True)
    cand_values = cand_values[order]
    if not np.array_equal(ref_ids["dx"].to_numpy(), cand_ids["dx"].to_numpy()):
        raise StabilityInputError("Reference and candidate lesion labels differ")
    return ref_values, cand_values, ref_ids


def coefficient_stability_from_lesions(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = _matrix(reference, "reference lesion coefficients")
    candidate = _matrix(candidate, "candidate lesion coefficients")
    if reference.shape != candidate.shape:
        raise StabilityInputError("Lesion coefficient matrices must have identical shape")
    per_archetype: list[dict] = []
    for index in range(reference.shape[1]):
        ref_column = reference[:, index]
        cand_column = candidate[:, index]
        degenerate = (
            np.std(ref_column) <= DEGENERATE_STD_THRESHOLD
            or np.std(cand_column) <= DEGENERATE_STD_THRESHOLD
        )
        correlation = None if degenerate else float(np.corrcoef(ref_column, cand_column)[0, 1])
        per_archetype.append(
            {
                "archetype": index + 1,
                "pearson_correlation": correlation,
                "correlation_defined": not degenerate,
                "mae": float(np.mean(np.abs(ref_column - cand_column))),
            }
        )
    l1 = np.sum(np.abs(reference - candidate), axis=1)
    dominant = np.argmax(reference, axis=1) == np.argmax(candidate, axis=1)
    return {
        "lesions": int(len(reference)),
        "per_archetype": per_archetype,
        "coefficient_vector_l1_mean": float(l1.mean()),
        "coefficient_vector_l1_by_lesion": l1,
        "dominant_archetype_agreement": float(dominant.mean()),
        "dominant_agreement_by_lesion": dominant,
    }


def coefficient_stability(
    reference: np.ndarray,
    reference_mapping: pd.DataFrame,
    aligned_candidate: np.ndarray,
    candidate_mapping: pd.DataFrame,
) -> dict:
    ref, cand, identities = paired_lesion_coefficients(
        reference, reference_mapping, aligned_candidate, candidate_mapping
    )
    result = coefficient_stability_from_lesions(ref, cand)
    result["lesion_mapping"] = identities
    return result


def coefficient_weighted_class_distribution(
    coefficients: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str] = EXPECTED_CLASSES,
) -> np.ndarray:
    coefficients = _matrix(coefficients, "coefficients")
    labels = np.asarray(labels, dtype=str)
    if len(coefficients) != len(labels):
        raise StabilityInputError("Coefficient and label rows differ")
    distributions = np.zeros((coefficients.shape[1], len(classes)), dtype=np.float64)
    for class_index, diagnosis in enumerate(classes):
        distributions[:, class_index] = coefficients[labels == diagnosis].sum(axis=0)
    totals = distributions.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise StabilityInputError("An archetype has no coefficient mass")
    return distributions / totals


def jensen_shannon_divergence_base2(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise StabilityInputError("Jensen-Shannon inputs must be same-length vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left < 0) or np.any(right < 0):
        raise StabilityInputError("Jensen-Shannon inputs must be finite and nonnegative")
    if left.sum() <= 0 or right.sum() <= 0:
        raise StabilityInputError("Jensen-Shannon inputs must each have positive mass")
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)

    def kl(values: np.ndarray, target: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / target[mask])))

    return 0.5 * kl(left, middle) + 0.5 * kl(right, middle)


def semantic_stability(
    reference_coefficients: np.ndarray,
    reference_labels: Sequence[str],
    aligned_candidate_coefficients: np.ndarray,
    candidate_labels: Sequence[str],
    classes: Sequence[str] = EXPECTED_CLASSES,
) -> dict:
    ref_distribution = coefficient_weighted_class_distribution(
        reference_coefficients, reference_labels, classes
    )
    cand_distribution = coefficient_weighted_class_distribution(
        aligned_candidate_coefficients, candidate_labels, classes
    )
    if ref_distribution.shape != cand_distribution.shape:
        raise StabilityInputError("Semantic distribution shapes differ")
    divergences = [
        jensen_shannon_divergence_base2(ref_distribution[i], cand_distribution[i])
        for i in range(ref_distribution.shape[0])
    ]
    return {
        "classes": list(classes),
        "reference_distribution": ref_distribution,
        "candidate_distribution": cand_distribution,
        "jensen_shannon_divergence_base2": np.asarray(divergences),
        "mean_jensen_shannon_divergence_base2": float(np.mean(divergences)),
    }


def nearest_real_top_n(
    latent_means: np.ndarray,
    z_fixed: np.ndarray,
    image_ids: Sequence[str],
    labels: Sequence[str],
    top_n: int = NEAREST_REAL_TOP_N,
) -> list[list[dict]]:
    latent_means = _matrix(latent_means, "latent_means")
    z_fixed = _matrix(z_fixed, "z_fixed")
    image_ids = np.asarray(image_ids, dtype=str)
    labels = np.asarray(labels, dtype=str)
    if len(latent_means) != len(image_ids) or len(image_ids) != len(labels):
        raise StabilityInputError("Nearest-real arrays have inconsistent row counts")
    if latent_means.shape[1] != z_fixed.shape[1]:
        raise StabilityInputError("Latent and fixed-simplex dimensions differ")
    if top_n <= 0 or top_n > len(latent_means):
        raise StabilityInputError("top_n is outside the available test population")
    result: list[list[dict]] = []
    for archetype, vertex in enumerate(z_fixed, start=1):
        distances = np.linalg.norm(latent_means - vertex, axis=1)
        order = np.argsort(distances, kind="stable")[:top_n]
        result.append(
            [
                {
                    "archetype": archetype,
                    "rank": rank,
                    "image_id": image_ids[index],
                    "dx": labels[index],
                    "distance": float(distances[index]),
                }
                for rank, index in enumerate(order, start=1)
            ]
        )
    return result


def compare_nearest_real(
    reference: list[list[dict]],
    candidate_raw: list[list[dict]],
    candidate_column_for_reference: Sequence[int],
    classes: Sequence[str] = EXPECTED_CLASSES,
) -> list[dict]:
    if len(reference) != len(candidate_raw):
        raise StabilityInputError("Nearest-real K/permutation mismatch")
    permutation = _permutation(candidate_column_for_reference, len(reference))
    rows: list[dict] = []
    for ref_index, candidate_index in enumerate(permutation):
        ref_rows = reference[ref_index]
        cand_rows = candidate_raw[int(candidate_index)]
        ref_ids = {row["image_id"] for row in ref_rows}
        cand_ids = {row["image_id"] for row in cand_rows}
        union = ref_ids | cand_ids
        ref_counts = np.asarray([sum(row["dx"] == cls for row in ref_rows) for cls in classes], dtype=float)
        cand_counts = np.asarray([sum(row["dx"] == cls for row in cand_rows) for cls in classes], dtype=float)
        rows.append(
            {
                "reference_archetype": ref_index + 1,
                "candidate_raw_archetype": int(candidate_index) + 1,
                "image_id_jaccard": float(len(ref_ids & cand_ids) / len(union)),
                "diagnosis_jensen_shannon_divergence_base2": jensen_shannon_divergence_base2(
                    ref_counts, cand_counts
                ),
            }
        )
    return rows


def visual_stability(
    reference_rgb: np.ndarray,
    candidate_rgb_raw: np.ndarray,
    candidate_column_for_reference: Sequence[int],
    data_range: float = SSIM_DATA_RANGE,
) -> list[dict]:
    import tensorflow as tf

    reference_rgb = np.asarray(reference_rgb, dtype=np.float32)
    candidate_rgb_raw = np.asarray(candidate_rgb_raw, dtype=np.float32)
    if reference_rgb.ndim != 4 or reference_rgb.shape[1:] != (128, 128, 3):
        raise StabilityInputError(f"Reference decoded images must have shape [K,128,128,3]; got {reference_rgb.shape}")
    if candidate_rgb_raw.shape != reference_rgb.shape:
        raise StabilityInputError("Candidate decoded-image shape or permutation differs")
    if not np.isfinite(reference_rgb).all() or not np.isfinite(candidate_rgb_raw).all():
        raise StabilityInputError("Decoded images must be finite")
    if data_range <= 0:
        raise StabilityInputError("Decoded-image data range must be positive")
    if (
        np.min(reference_rgb) < -1e-6
        or np.min(candidate_rgb_raw) < -1e-6
        or np.max(reference_rgb) > data_range + 1e-6
        or np.max(candidate_rgb_raw) > data_range + 1e-6
    ):
        raise StabilityInputError("Decoded images fall outside the declared data range")
    permutation = _permutation(candidate_column_for_reference, len(reference_rgb))
    candidate_aligned = candidate_rgb_raw[permutation]
    ssim = tf.image.ssim(reference_rgb, candidate_aligned, max_val=float(data_range)).numpy()
    mse = np.mean((reference_rgb - candidate_aligned) ** 2, axis=(1, 2, 3))
    return [
        {
            "reference_archetype": index + 1,
            "candidate_raw_archetype": int(permutation[index]) + 1,
            "rgb_ssim": float(ssim[index]),
            "pixel_mse": float(mse[index]),
            "data_range": float(data_range),
        }
        for index in range(len(reference_rgb))
    ]


def lesion_bootstrap_coefficient_stability(
    reference_lesion_coefficients: np.ndarray,
    candidate_lesion_coefficients: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    reference = _matrix(reference_lesion_coefficients, "reference lesion coefficients")
    candidate = _matrix(candidate_lesion_coefficients, "candidate lesion coefficients")
    if reference.shape != candidate.shape or len(reference) < 2:
        raise StabilityInputError("Bootstrap inputs must have identical shape and at least two lesions")
    if resamples <= 0:
        raise StabilityInputError("resamples must be positive")
    rng = np.random.default_rng(seed)
    metric_names = ["coefficient_vector_l1_mean", "dominant_archetype_agreement"]
    samples = {name: np.empty(resamples, dtype=float) for name in metric_names}
    for draw in range(resamples):
        indices = rng.integers(0, len(reference), size=len(reference))
        ref_draw = reference[indices]
        cand_draw = candidate[indices]
        samples["coefficient_vector_l1_mean"][draw] = np.mean(
            np.sum(np.abs(ref_draw - cand_draw), axis=1)
        )
        samples["dominant_archetype_agreement"][draw] = np.mean(
            np.argmax(ref_draw, axis=1) == np.argmax(cand_draw, axis=1)
        )
    return {
        "unit": "lesion_id",
        "resamples": resamples,
        "seed": seed,
        "percentile_interval": [2.5, 97.5],
        "metrics": {
            name: {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }
            for name, values in samples.items()
        },
    }


def summarize_seed_comparisons(values: Sequence[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) != 4 or not np.isfinite(values).all():
        raise StabilityInputError("Exactly four finite seed-113-to-candidate values are required")
    return {
        "raw": values.tolist(),
        "mean": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "inferential_interval_across_seed_comparisons": None,
    }
