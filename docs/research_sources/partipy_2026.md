# ParTIpy (2026)

## Source

“ParTIpy: a scalable framework for archetypal analysis and Pareto task inference,” *Molecular Systems Biology* (2026), DOI `10.1038/s44320-026-00209-6`. [Publisher article](https://link.springer.com/article/10.1038/s44320-026-00209-6).

## Relevant ideas

ParTIpy emphasizes that archetypal-analysis optimization is non-convex and sensitive to initialization. It benchmarks multiple seeds/initializations, uses bootstrap resampling for robustness, and treats archetype labels as non-identifiable across runs.

For bootstrap or implementation comparisons, it aligns learned archetype locations to a reference using Hungarian matching with pairwise Euclidean location costs. Stability is then summarized by positional variance. Its coreset evaluation similarly uses multiple random seeds and aligned solutions.

The framework also includes a bounded vector `alpha` with elementwise bounds `[1-delta, 1+delta]` in a coreset-weighted classical AA objective, demonstrating a controlled relaxation concept rather than unconstrained movement.

## Amortization-gap motivation

ParTIpy performs direct optimization of per-observation AA coefficients and highlights optimization/initialization sensitivity. DeepAA instead gains fast inference from an encoder that amortizes coordinate inference. Comparing encoder output with frozen-model, per-instance direct optimization is therefore a relevant DeepAA reliability question.

This comparison is a project-level transfer and interpretation. The inspected ParTIpy article does **not** itself present a DeepAA amortization-gap experiment, so it must not be cited as if it had measured that gap.

## What transfers to DeepAA

- Multi-run and bootstrap thinking for robustness.
- Explicit handling of label permutation.
- Hungarian assignment as an alignment mechanism.
- Joint consideration of reconstruction, model complexity, and stability when selecting `K`.
- Bounded relaxation as a disciplined alternative to arbitrary free archetype movement.
- Reporting runtime and solution quality across repeated runs.

## What does not transfer directly

- ParTIpy learns archetype locations from data in a PCA/integrated feature space. DeepAA's `Z_fixed` vertices are predetermined.
- Euclidean distances between fixed vertices contain no run-specific semantic information; identical geometry can decode to different semantics.
- ParTIpy's transcriptomic preprocessing, distance-weighted gene profiles, enrichment tests, and coreset claims do not transfer to images without a new method and validation.
- Its bounded `alpha` appears in a classical/coreset AA objective and is not automatically equivalent to `Z_delta = diag(s) Z_fixed` in a neural decoder.
- Direct classical AA coefficients and a neural encoder's variational/local inference parameters are not interchangeable.

## DeepAA-specific alignment consequence

For a shared validation/reference sample set, construct each archetype's activation signature from coefficient columns and use a cost such as

```text
cost(i, j) = 1 - corr(a_reference[:, i], a_run[:, j])
```

before Hungarian assignment. Evaluate final stability on held-out test samples. Constant or near-constant columns, ties, sign conventions, and missing usage require a documented fallback before implementation.
