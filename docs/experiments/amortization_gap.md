# Amortization-gap protocol

Status: **PROPOSED; not implemented; no results**.

## Question

How suboptimal are encoder-produced inference parameters relative to per-instance optimization under the same frozen DeepAA model on held-out HAM10000 images?

## Definition

For a minimized per-image objective `L`:

```text
gap(x) = L_amortized(x) - L_locally_refined(x)
relative_gap(x) = gap(x) / max(|L_amortized(x)|, epsilon)
```

Report the signed gap as well as any clipped improvement summary so regressions are visible.

## Protocol

1. Load an approved checkpoint and freeze every global/model parameter.
2. Encode held-out image `x` and save the original inference parameters, objective components, coefficients, reconstruction, prediction, and runtime.
3. Initialize only that image's local inference parameters from the encoder.
4. Optimize those local parameters for depths `0, 1, 3, 5`; add deeper settings only through a recorded decision.
5. Preserve simplex validity, provisionally with unconstrained logits mapped through softmax when compatible with the authoritative model.
6. Recompute the same metrics at each depth without changing model weights.
7. Aggregate over all selected held-out samples and repeat checkpoints/seeds as approved.

## No-test-label-leakage rule

The true held-out diagnosis label must not appear anywhere in the local optimization objective, stopping rule, step selection, or hyperparameter tuning. Using it would be label-guided test-time optimization, not a clean amortization-gap measurement. Diagnosis labels may be used after refinement to measure prediction changes.

## Metrics

- Total objective and every objective component.
- Absolute and relative gap.
- Reconstruction MSE and perceptual difference if approved.
- Coefficient L1/L2 displacement, entropy change, dominant-archetype change, and boundary behavior.
- Predicted class/probability change without using the label during optimization.
- Encoder and per-step wall-clock runtime with synchronized device timing, warm-up, and batch-size policy recorded.
- Failure/non-finite rate and samples for which refinement worsens the objective.

## Open choices

The authoritative local variational parameters are unknown because the located HAM code is deterministic. The exact objective, logits mapping, optimizer, step size, clipping, random sampling policy, stopping criterion, and timing protocol require approval after Gate 0.

