# Archetype stability protocol

Status: **PROPOSED; not implemented; no results**.

## Question

Do independent DeepAA training runs recover the same meaningful archetypal semantics after accounting for permutation of archetype identities?

## Preconditions

- Gate 0 baseline reproduced.
- One persisted lesion-grouped train/validation/test split.
- Approved seed list and canonical configuration for each `K`.
- Deterministic extraction of coefficients, decoded archetypes, nearest images, and enrichment summaries.

## Protocol

1. Train the same configuration independently for every approved seed.
2. Choose an approved reference run or consensus reference without inspecting test outcomes.
3. On the fixed validation/reference set, form coefficient activation signatures, one sample-length vector per archetype per run.
4. Compute pairwise costs, initially `1 - Pearson correlation`, between reference and run signatures.
5. Handle constant/near-constant signatures with a pre-approved fallback; record raw costs and the assignment.
6. Solve the one-to-one assignment with the Hungarian algorithm.
7. Apply the learned permutation to all run artifacts.
8. On the untouched test set, calculate stability metrics and uncertainty across runs.

Do not match `A3` to `A3` by index. Do not use Euclidean distance between `Z_fixed` vertices as the semantic cost.

## Held-out evaluation

- Decoded-archetype visual similarity using approved pixel/perceptual metrics plus blinded qualitative review.
- Nearest-real-image overlap and rank agreement.
- Same-lesion coefficient correlation, absolute difference, L1/L2 displacement, and dominant-archetype agreement.
- Archetype usage frequency and inactive-archetype rate.
- Diagnosis enrichment/purity stability with uncertainty; labels are for evaluation, not alignment.
- Reconstruction and classification variation across seeds.

## Analysis plan

Report per-archetype and aggregate distributions, assignment costs, confidence intervals or bootstrap intervals, and failures. Compare raw-index results with aligned results only to demonstrate why alignment matters. Stratify by `K`; do not pool incompatible coordinate dimensions.

## Risks/open choices

- Reference-run selection can affect assignments.
- Pearson correlation is undefined for constant activation columns.
- Rarely used archetypes may align unreliably.
- Decoder symmetries can yield visually similar but activation-different solutions.
- Required seed count, similarity metrics, uncertainty estimator, and consensus procedure remain open decisions.

