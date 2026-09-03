# Controlled simplex-relaxation protocol

Status: **PROPOSED; blocked on Gate 3 design approval; no results**.

## Formulation

Baseline:

```text
mu(x) = a Z_fixed
```

Introduce global bounded radial scales:

```text
s_k in [1 - delta, 1 + delta]
Z_delta = diag(s) Z_fixed
mu(x) = a Z_delta
```

Required recovery condition: at `delta = 0`, the implementation and objective must reduce exactly to the approved conference baseline.

Candidate ablation:

```text
delta in {0, 0.1, 0.25, 0.5}
```

This is controlled radial relaxation, not arbitrary free movement of archetypes.

## Planned comparisons

Use identical persisted splits, seed sets, `K` values, training budgets, selection rules, and evaluation code. Record learned scales and whether bounds are active.

Metrics:

- reconstruction and full objective components;
- classification with a leakage-free train/validation/test protocol;
- coefficient entropy, sparsity, active archetypes, and usage balance;
- simplex adherence/boundary behavior;
- semantic enrichment/purity;
- decoded archetypes and nearest real images;
- permutation-aligned multi-seed stability;
- parameter/runtime change.

## Open regularizer decision — do not implement yet

The existing DeepAA archetype regularizer may target `Z_fixed`. If the generative geometry uses `Z_delta` while the regularizer still pulls toward `Z_fixed`, it may cancel or distort the intended relaxation. ChatGPT and the senior developer must approve whether the target changes, the regularizer changes, or another formulation is used. The decision must state the objective mathematically and demonstrate exact `delta = 0` recovery.

Other open decisions include scale parameterization (bounded sigmoid versus projection), initialization, whether a scale prior/penalty is needed, identifiability constraints, and how the scale interacts with latent variance.

## Validity checks

- Unit/equivalence test at `delta = 0` for forward outputs and loss components.
- Bounds always satisfied, gradients finite, and scales saved in checkpoints.
- No unintentional change to coefficient simplex constraints.
- Ablation conclusions based on repeated seeds, not one run.

