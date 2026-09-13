# Decision log

Allowed statuses: **PROPOSED**, **ACCEPTED**, **REJECTED**, **SUPERSEDED**.

## D-001 — One reliability-centered journal story

- Date: 2026-09-02
- Status: ACCEPTED
- Decision: Frame stability, amortization gap, controlled relaxation, and counterfactuals as connected tests/uses of reliable simplex coordinates, not four independent projects.
- Basis: Explicit senior-developer project direction.

## D-002 — Activation-signature alignment

- Date: 2026-09-02
- Status: SUPERSEDED
- Decision: Align independent runs with Hungarian assignment using validation/reference coefficient-activation correlations.
- Rationale: `Z_fixed` locations are predetermined and contain no run-specific semantic identity.
- Superseded by: D-015 and D-016, which accept the reference rule, separation, threshold, and fallback.
- Reconciliation audit (2026-09-13): Git commit `7911e8643495a2f34820ca0adedbea1f322ab8da` (2026-09-03) changed this entry from PROPOSED to SUPERSEDED and recorded D-015/D-016 as ACCEPTED. The older proposal is not the current authority; no new or backdated approval is inferred. Existing `research/code/alignment.py` and the alignment/evaluation CLIs implement those accepted decisions. The first K5 seed113-versus-seed211 comparison executed on 2026-09-13 after artifact validation; see the experiment registry. D-002 remains SUPERSEDED.

## D-003 — No diagnosis-label signal in amortization refinement

- Date: 2026-09-02
- Status: ACCEPTED
- Decision: Never use the true held-out diagnosis to optimize local inference parameters or select refinement depth.
- Rationale: Preserve a scientifically clean inference-gap measurement.

## D-004 — Bounded radial simplex relaxation

- Date: 2026-09-02
- Status: PROPOSED
- Decision: Study `Z_delta = diag(s) Z_fixed`, `s_k in [1-delta,1+delta]`, initially for `delta = {0, 0.1, 0.25, 0.5}`.
- Blocker: regularizer compatibility and exact baseline recovery are not approved.

## D-005 — Counterfactuals in simplex coordinates

- Date: 2026-09-02
- Status: PROPOSED
- Decision: Optimize target loss plus coefficient-space L1 distance over valid simplex coefficients; consider target-class prototypes later.

## D-006 — Medical counterfactual language

- Date: 2026-09-02
- Status: ACCEPTED
- Decision: Describe generated counterfactuals only as explanations of model behavior, never clinical causality, lesion progression, treatment, or patient advice.

## D-007 — Lesion-grouped split

- Date: 2026-09-02
- Status: ACCEPTED
- Decision: Replace image-level splitting with a persisted lesion-grouped, diagnosis-stratified split for journal experiments.
- Rationale: multiple HAM10000 images can share a `lesion_id`; cross-partition lesion overlap risks leakage.
- Implemented by: D-013 and immutable `research/splits/journal_v1` manifests.

## D-008 — Authoritative baseline identity

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Treat `https://github.com/Amy007-ali/Deep-Archetypal-Analysis-on-HAM10000.git` as the authoritative conference code repository; do not substitute the materially different deterministic PyTorch AANet project.
- Basis: The senior developer explicitly identified the GitHub repository as authoritative; inspection confirms the architecture described by the conference abstract and exact rounded published-result matches.

## D-009 — Working conference baseline commit

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Use HEAD `29acec505161530f9ea811534b05dd52d8df9899` as the working conference baseline for preservation and evaluation-only reproduction.
- Evidence: Senior-developer direction; HEAD completes the May K=3/5/7/9 result series, every published rounded value matches, all four checkpoints reproduce committed metrics, and the source/README are unchanged from `661cc7f` through HEAD.
- Limitation: This is high-confidence working provenance, not formal certainty. No conference tag/release, submitted archive, or author confirmation binds the submission to this commit.

## D-010 — Canonical documentation location

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: The repository-local `docs/` directory is the canonical project documentation location from Gate 0 onward.
- Constraint: Preserve the external `C:\Users\UserPc\Desktop\ai\docs` copy until separately authorized to remove it.

## D-011 — Historical split preservation

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Preserve the conference image-level split exactly for historical reproduction and audit its `lesion_id` overlap; do not retroactively replace it with a grouped split.
- Evidence: Exact test order and batch size 32 reproduce every committed checkpoint metric. The manifests contain 8,013 train, 1,001 validation, and 1,001 test images; 725 lesion IDs cross at least one split boundary.
- Future boundary: A lesion-grouped split remains a separate proposed journal protocol and must not rewrite conference claims.

## D-012 — Final May entropy behavior

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: The working May conference baseline has no coefficient-entropy stabilization or temperature objective during training. Entropy is evaluation-only.
- Evidence: Authoritative May source and January-to-May Git diff.

## D-013 — Accepted journal split v1

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: For all new research, use `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=113)` with `lesion_id` groups and `dx` stratification; assign fold 0 to test, fold 1 to validation, and folds 2–9 to train.
- Anti-cherry-picking rule: Do not search seeds or reassign folds for better class balance.
- Evidence: Generated manifests cover all 10,015 images exactly once, contain all seven classes in every partition, and have zero pairwise lesion overlap.
- Review state: Accepted explicitly by the senior developer on 2026-09-03.
- Immutability: Never mutate or overwrite `journal_v1`. Any replacement requires a separately reviewed `journal_v2` directory and protocol amendment.

## D-014 — Stability v1 primary scope

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Restrict the first stability experiment to K=`{5,7}` and seeds=`{113,211,307,401,503}`, for ten primary runs on one fixed journal_v1 split.
- Configuration: Preserve the final conference architecture and objective; explicitly use 128×128 RGB, batch 32, 50 epochs, Adam at 0.001, loss weights `1/1/0.01/1`, and final-epoch checkpoints.
- Constraint: No additional seeds or K values belong to the primary v1 experiment without a dated protocol amendment.

## D-015 — Predetermined stability reference and data separation

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Use seed 113 as the predetermined reference independently for each K. Learn Hungarian permutations from validation coefficient-activation signatures with cost `1 - Pearson correlation`; measure final stability only on test.
- Prohibited criterion: Do not align semantics using Euclidean distance between identical fixed simplex vertices.
- Preservation: Retain raw archetype numbering, cost matrices, learned permutations, and aligned derivatives.
- Separate future analysis: Consensus matching may be studied later but must not silently replace this primary reference rule.

## D-016 — Deterministic fallback for degenerate alignment columns

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Treat a validation coefficient column as degenerate when its population standard deviation is at most `1e-8`. For a pair in which either column is degenerate, use `2 * MAE` as the Hungarian cost; otherwise use `1 - Pearson correlation`.
- Rationale: The fallback is deterministic and finite for constant columns, and the factor of two gives the same bounded `[0,2]` scale as correlation cost for simplex coefficients.
- Preservation: Save both the full cost matrix and the per-cell method matrix.

## D-017 — Stability visual metric

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Compare aligned decoded archetypes using RGB SSIM as the primary metric with explicit data range 1.0; report pixel MSE as a secondary metric.
- Constraint: Decode each run's raw fixed vertices deterministically and apply the validation-learned permutation only to derived comparisons.

## D-018 — Stability aggregation and uncertainty

- Date: 2026-09-03
- Status: ACCEPTED
- Decision: Within each K, retain all four seed-113-versus-candidate values and summarize them using arithmetic mean, sample standard deviation, minimum, and maximum. Do not infer a confidence interval from four training seeds.
- Lesion uncertainty: For lesion-level coefficient-vector L1 distance and dominant-archetype agreement, use 1,000 paired lesion bootstrap resamples with replacement, seed 2026, and a two-sided percentile 95% interval.
