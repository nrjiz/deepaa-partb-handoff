# Archetype stability protocol

Status: **PROPOSED AND PRE-REGISTERED; not implemented; no results**.

Canonical detailed protocol: `research/protocols/archetype_stability_v1.md`; machine-readable record: `research/configs/archetype_stability_v1.json`.

## Question

Do independent DeepAA training runs recover the same meaningful archetypal semantics after accounting for permutation of archetype identities?

## Preconditions

- Gate 0 baseline reproduced: PASS WITH DOCUMENTED LIMITATIONS.
- Proposed `journal_v1` lesion-grouped split generated and invariant-checked; senior-developer acceptance remains pending.
- Fixed primary scope: K=`{5,7}`, seeds=`{113,211,307,401,503}`, ten runs total.
- Deterministic extraction of coefficients, decoded archetypes, nearest images, and enrichment summaries.

## Protocol

1. Train the same configuration independently for every approved seed.
2. Use seed 113 as the predetermined reference independently for each K; do not choose by appearance or performance.
3. On the fixed validation/reference set, form coefficient activation signatures, one sample-length vector per archetype per run.
4. Compute pairwise costs, initially `1 - Pearson correlation`, between reference and run signatures.
5. Handle constant/near-constant signatures with a pre-approved fallback; record raw costs and the assignment.
6. Solve the one-to-one assignment with the Hungarian algorithm.
7. Apply the learned permutation to all run artifacts.
8. On the untouched test set, calculate stability metrics and uncertainty across runs.

Do not match `A3` to `A3` by index. Do not use Euclidean distance between `Z_fixed` vertices as the semantic cost.

## Held-out evaluation

- Decoded-archetype visual similarity after matching; the exact image metric remains open.
- Top-10 nearest-test-image Jaccard overlap and diagnosis-distribution consistency.
- Lesion-aggregated matched coefficient correlation, absolute difference, L1 displacement, and dominant-archetype agreement.
- Archetype usage frequency and inactive-archetype rate.
- Coefficient-weighted `P(dx | archetype)` stability using base-2 Jensen–Shannon divergence; labels are for test evaluation, not alignment.
- Reconstruction and classification variation across seeds.

## Analysis plan

Report per-archetype and aggregate distributions, assignment costs, confidence intervals or bootstrap intervals, and failures. Compare raw-index results with aligned results only to demonstrate why alignment matters. Stratify by `K`; do not pool incompatible coordinate dimensions.

## Risks/open choices

- Pearson correlation is undefined for constant activation columns.
- Rarely used archetypes may align unreliably.
- Decoder symmetries can yield visually similar but activation-different solutions.
- The inactive-column threshold/fallback, exact visual-similarity metric, uncertainty estimator, and final infrastructure review remain open decisions.
- Consensus matching, if later explored, is a separate analysis and does not replace the seed-113 primary reference.
