# Decision log

Allowed statuses: **PROPOSED**, **ACCEPTED**, **REJECTED**, **SUPERSEDED**.

## D-001 — One reliability-centered journal story

- Date: 2026-09-02
- Status: ACCEPTED
- Decision: Frame stability, amortization gap, controlled relaxation, and counterfactuals as connected tests/uses of reliable simplex coordinates, not four independent projects.
- Basis: Explicit senior-developer project direction.

## D-002 — Activation-signature alignment

- Date: 2026-09-02
- Status: PROPOSED
- Decision: Align independent runs with Hungarian assignment using validation/reference coefficient-activation correlations.
- Rationale: `Z_fixed` locations are predetermined and contain no run-specific semantic identity.
- Open: fallback costs, reference selection, and constant columns.

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
- Status: PROPOSED
- Decision: Replace image-level splitting with a persisted lesion-grouped, diagnosis-stratified split for journal experiments.
- Rationale: multiple HAM10000 images can share a `lesion_id`; cross-partition lesion overlap risks leakage.

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

## D-013 — Proposed journal split v1

- Date: 2026-09-03
- Status: PROPOSED
- Decision: For all new research, use `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=113)` with `lesion_id` groups and `dx` stratification; assign fold 0 to test, fold 1 to validation, and folds 2–9 to train.
- Anti-cherry-picking rule: Do not search seeds or reassign folds for better class balance.
- Evidence: Generated manifests cover all 10,015 images exactly once, contain all seven classes in every partition, and have zero pairwise lesion overlap.
- Review state: `PROPOSED JOURNAL SPLIT V1`; senior-developer acceptance is required before training.

## D-014 — Stability v1 primary scope

- Date: 2026-09-03
- Status: PROPOSED
- Decision: Restrict the first stability experiment to K=`{5,7}` and seeds=`{113,211,307,401,503}`, for ten primary runs on one fixed journal_v1 split.
- Configuration: Preserve the final conference architecture and objective; explicitly use 128×128 RGB, batch 32, 50 epochs, Adam at 0.001, loss weights `1/1/0.01/1`, and final-epoch checkpoints.
- Constraint: No additional seeds or K values belong to the primary v1 experiment without a dated protocol amendment.

## D-015 — Predetermined stability reference and data separation

- Date: 2026-09-03
- Status: PROPOSED
- Decision: Use seed 113 as the predetermined reference independently for each K. Learn Hungarian permutations from validation coefficient-activation signatures with cost `1 - Pearson correlation`; measure final stability only on test.
- Prohibited criterion: Do not align semantics using Euclidean distance between identical fixed simplex vertices.
- Preservation: Retain raw archetype numbering, cost matrices, learned permutations, and aligned derivatives.
- Separate future analysis: Consensus matching may be studied later but must not silently replace this primary reference rule.

## D-016 — Unstable correlation columns remain unresolved

- Date: 2026-09-03
- Status: PROPOSED
- Decision: Do not silently substitute a matching cost when a coefficient column is constant or nearly inactive. Preserve and flag the affected matrix, then stop that comparison until a threshold and fallback are reviewed.
