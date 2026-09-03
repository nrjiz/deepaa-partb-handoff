# Simplex-coordinate counterfactual protocol

Status: **PROPOSED; no implementation; no results**.

## Goal

Explain a frozen classifier's behavior by finding a changed archetypal mixture that produces a target prediction while remaining close to the original mixture.

## Initial formulation

For original coefficients `a`, optimize:

```text
a_cf in Delta^K

target_loss(classifier(decoder(a_cf Z)), target_class)
    + lambda * ||a_cf - a||_1

x_cf = decoder(a_cf Z)
```

Parameterize `a_cf = softmax(u)` unless the authoritative baseline requires another approved simplex map. Initialize `u` from the original coefficients with a numerically safe inverse/approximation.

## Initial experiment

1. Freeze encoder, geometry, decoder, and classifier.
2. Select test queries and target classes by a policy fixed without cherry-picking outcomes.
3. Optimize only `u` for an approved step budget and `lambda` grid chosen on validation data.
4. Save the full trajectory, coefficient vector, decoded image, prediction probabilities, objective terms, and runtime.
5. Evaluate on held-out test queries.

## Evaluation

- Validity/success rate at a predeclared target-probability threshold.
- Coefficient L1 distance, number of materially changed coordinates, entropy change, and dominant-archetype change.
- Image-space and approved perceptual distances.
- Reconstruction consistency and plausibility/quality review.
- Runtime and convergence/failure rate.
- Stability across initialization, seeds/checkpoints, and nearby `lambda` values.
- Readable coefficient-change summaries for every displayed case.

## Later optional extension

Add distance to a target-class prototype defined in archetypal-coordinate space. Prototype construction must use training data only; validation selects its weight. This is not part of the initial implementation.

## Safeguards and limitations

- Never optimize using the query's true diagnosis as a hidden plausibility/reconstruction signal.
- Clearly distinguish target selection from ground-truth evaluation.
- Valid simplex coefficients do not guarantee a realistic lesion image.
- Classifier success may exploit artifacts.
- Explanations are model-specific and may change across checkpoints.
- Do not describe generated transitions as disease evolution, biological progression, treatment, patient recourse, or causal medicine.

