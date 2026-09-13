# Archetype stability protocol

Status: **ACCEPTED AND PRE-REGISTERED; implemented; no primary results**.

Execution update (2026-09-13): the status above records the original implementation checkpoint. Two primary K5 runs (seeds113/211) and their first validation-only Hungarian/test comparison are now completed and validated; see the dated experiment registry. One comparison does not complete the frozen multi-seed study. No methodological change accompanies this status update.

Canonical protocol: `research/protocols/archetype_stability_v1.md`. Machine-readable record: `research/configs/archetype_stability_v1.json`.

## Fixed design

- Accepted immutable `journal_v1` lesion-grouped split; any replacement requires `journal_v2`.
- K=`{5,7}`, seeds=`{113,211,307,401,503}`, ten runs total.
- Seed 113 is the predetermined reference within each K.
- Final May model/objective and preprocessing are unchanged.
- Validation coefficients alone determine each Hungarian permutation; test data are evaluation-only.
- Ordinary matching cost is `1 - Pearson correlation`. If either column's population standard deviation is at most `1e-8`, the deterministic fallback is `2 * MAE`.
- Raw numbering and outputs are preserved; aligned outputs are separate derivatives.

## Held-out evaluation

- Lesion-mean coefficient correlations and MAE, coefficient-vector L1 distance, and dominant-archetype agreement.
- Coefficient-weighted diagnosis distributions compared with base-2 Jensen-Shannon divergence.
- Matched top-10 nearest-test-image Jaccard overlap and diagnosis-distribution consistency.
- Decoded-archetype RGB SSIM with data range 1.0 as primary visual similarity, with pixel MSE secondary.
- Every one of the four reference-candidate comparisons per K, plus arithmetic mean, sample standard deviation, minimum, and maximum. No confidence interval is inferred from four training seeds.
- Paired 1,000-resample lesion bootstrap, seed 2026, percentile 95% bounds for lesion-level vector L1 and dominant agreement.

## Implementation status

The manifest adapter, deterministic process controls, exact single-run and matrix launchers, authoritative-model parity checks, validation alignment, held-out metrics, overwrite protection, tests, and external-GPU instructions are implemented under `research/`. The matrix launcher is dry-run by default and the single-run launcher refuses a dirty primary worktree.

A completed tiny smoke test validates data-to-artifact plumbing and checkpoint reload only. It is explicitly marked `scientific_evidence: false`; its model metrics must not be reported as findings. No one of the ten primary runs has started.

Do not compare raw archetype IDs across runs, use test data for matching, mutate `journal_v1`, tune the frozen primary matrix after results, or overwrite a completed output directory.
