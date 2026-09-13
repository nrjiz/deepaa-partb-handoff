# Archetype stability experiment v1

Status: **ACCEPTED AND PRE-REGISTERED; NOT EXECUTED**.

Execution annotation (2026-09-13; methodology unchanged): the heading and no-run statement below preserve the original preregistration checkpoint. K5 seeds113/211 are now completed and validated, and one validation-aligned test comparison has executed. Current execution evidence is in `docs/experiments/experiment_registry.md` and `research/analysis/stability_v1/K5_113_vs_211_20260913/`. The full ten-run/eight-comparison study is not complete; this annotation does not amend its scientific rules.

## Scientific question

When DeepAA is retrained independently on the same leak-free HAM10000 split, does it recover consistent archetypal semantics after accounting for archetype permutation?

## Fixed primary scope

- Split: accepted `journal_v1`, identical across every run. Its manifests are immutable; a replacement requires a separately reviewed `journal_v2`.
- K values: `{5, 7}`; latent dimensions `{4, 6}` respectively.
- Training seeds: `{113, 211, 307, 401, 503}`.
- Total primary runs: 10.
- Predetermined reference: seed 113 independently within each K, never chosen by appearance or performance.
- A future consensus analysis may be added only as a separately labeled analysis; it must not replace the reference rule.

## Frozen model and training configuration

The experiment reuses the authoritative May conference model and objective without editing or copying them. The runner imports `daa_ham10000_interpretability_fixed_v2.py` only after verifying its SHA-256 is `6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d`.

Input is 128 x 128 RGB, batch size is 32, training lasts 50 epochs, and optimization uses Adam at learning rate 0.001 with no early stopping and a final-epoch checkpoint. Loss weights are reconstruction 1.0, classification 1.0, KL 0.01, and archetype/simplex 1.0. Coefficient-entropy training weight is 0.0. Preprocessing is JPEG RGB decode, bilinear resize, float32 division by 255, and no augmentation.

The manifest adapter is the sole source of membership and stable ordering. It verifies all three accepted hashes, expected counts, unique image membership, exact 10,015-image coverage, all seven classes, and zero pairwise lesion overlap before building a TensorFlow dataset. It never calls the historical image-level splitter.

## Randomness and reproducibility

Each run executes in a fresh process. Before TensorFlow import, set `PYTHONHASHSEED` and `TF_DETERMINISTIC_OPS`; then set Python, NumPy, and TensorFlow/Keras seeds. The training shuffle uses the run seed explicitly with `reshuffle_each_iteration=True`, while validation and test order remain fixed. TensorFlow deterministic operations are enabled where supported and success or warnings are recorded.

Every run records the exact command, configuration, git revision and cleanliness, source and manifest hashes, package/runtime/device details, warnings, dataset seed, and artifact inventory. Primary execution refuses a dirty worktree. Completed or nonempty output directories are never overwritten; an explicitly requested rerun receives a new timestamped sibling.

## Validation-only Hungarian alignment

For each K, extract validation coefficients for seed 113 and each candidate in identical manifest order. For reference archetype `i` and candidate archetype `j`:

```text
if population_std(ref[:, i]) > 1e-8 and population_std(candidate[:, j]) > 1e-8:
    cost(i,j) = 1 - PearsonCorrelation(ref[:, i], candidate[:, j])
else:
    cost(i,j) = 2 * mean_absolute_error(ref[:, i], candidate[:, j])
```

The factor of two keeps the bounded fallback on the same `[0, 2]` scale as correlation cost. Apply Hungarian linear assignment to the full K x K cost matrix. The implementation preserves the method used for every matrix cell, the cost matrix, the reference-to-candidate permutation, raw outputs, and separate aligned derivatives. Fixed-simplex vertex distance is prohibited as a semantic matching criterion.

## Held-out test evaluation

The test set never selects a permutation. After validation alignment:

1. Average coefficients across images sharing each test `lesion_id`. On the paired 748-lesion matrices, report matched-column Pearson correlation and MAE, full-vector L1 distance, and dominant-archetype agreement. Image-level variants are secondary diagnostics.
2. Compute coefficient-weighted `P(dx | archetype)` for each matched archetype and compare the seven-class distributions using base-2 Jensen-Shannon divergence.
3. Rank test images by Euclidean distance from `mu` to each run's corresponding raw `Z_fixed` vertex. Compare matched top-10 sets by Jaccard overlap and their diagnosis distributions by base-2 Jensen-Shannon divergence.
4. Decode every run's raw fixed vertices deterministically. After matching, use RGB SSIM with explicit data range 1.0 as the primary visual metric and pixel MSE as the secondary metric.

For each K, preserve all four seed-113-versus-candidate comparisons. Report every raw value plus arithmetic mean, sample standard deviation, minimum, and maximum. Do not construct an inferential confidence interval from only four training-seed comparisons. For lesion-level coefficient-vector L1 distance and dominant-archetype agreement, additionally report a paired lesion bootstrap interval using 1,000 resamples with replacement, seed 2026, and two-sided percentile 95% bounds.

## Implementation and output boundaries

- One run: `research/tools/run_stability_one.py`.
- Exact ten-run matrix: `research/tools/run_stability_matrix.py` (dry-run by default; `--execute` is explicit).
- Alignment: `research/tools/align_validation_coefficients.py`.
- Held-out evaluation: `research/tools/evaluate_stability_pair.py`.
- Primary artifacts: unique directories below `research/runs/stability_v1/` only.
- Smoke artifacts: unique directories below `research/smoke_tests/` only and labeled `scientific_evidence: false`.
- External GPU procedure: `research/external_gpu_execution.md`.

No primary stability run has been executed. The completed smoke test validates plumbing only and is not scientific evidence.

## Not permitted in v1

- Additional or replacement K values or seeds without a dated protocol amendment.
- Regenerating the split per training seed or mutating `journal_v1`.
- Reference selection based on appearance or metrics.
- Matching on test data or using fixed-vertex Euclidean distance for semantic alignment.
- Changing the baseline architecture, objective, preprocessing, or frozen training configuration.
- Treating smoke outputs as results or silently replacing the reference rule with consensus matching.
