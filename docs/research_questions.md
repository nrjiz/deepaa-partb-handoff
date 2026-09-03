# Research questions and hypotheses

## Main research problem

Can DeepAA's simplex-constrained archetypal representation for medical images be made reliable and robust while preserving interpretability and fast inference?

## Connected subquestions

### RQ1 — Stability and reproducibility

After accounting for permutation of archetype identities, how stable are decoded archetypes, nearest real examples, per-lesion coefficients, usage, and semantic enrichment across independent training seeds?

### RQ2 — Amortized inference reliability

On held-out images, how much does the frozen model objective improve when local inference parameters initialized by the encoder are optimized directly, and what accuracy/runtime trade-off appears after 1, 3, and 5 steps?

### RQ3 — Geometric robustness

Does bounded radial scaling of fixed simplex vertices improve reconstruction, prediction, coordinate use, semantics, or stability while preserving the interpretable simplex structure and recovering the baseline at `delta = 0`?

### RQ4 — Coordinate-level model explanation

Can counterfactual search in the validated simplex-coordinate space produce successful, sparse, plausible prediction changes that are easier to express as archetype-weight changes than opaque latent movements?

## Hypotheses — not findings

- **H1:** After activation-signature alignment, some archetypes will be more stable than raw index comparisons suggest, while stability will vary by `K` and semantic support.
- **H2:** Local label-free refinement will reduce the per-image objective relative to encoder-only inference, with diminishing returns as refinement depth increases.
- **H3:** A small nonzero `delta` may improve fit or coordinate utilization without destroying semantics; larger flexibility may reduce stability or interpretability.
- **H4:** Simplex-coordinate counterfactuals will permit sparse, human-readable coefficient changes, but validity and visual plausibility will depend on decoder and classifier quality.
- **H5:** Reliability dimensions will interact: unstable archetypes or a large amortization gap will weaken the meaning of coordinate-level counterfactuals.

These hypotheses are preregistered expectations. They must be retained, revised with dated justification, or rejected based on held-out evidence; they must never be written as completed results before experiments run.

