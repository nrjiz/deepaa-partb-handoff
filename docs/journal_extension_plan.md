# Journal extension plan

## One coherent story

The conference asked:

> Can DeepAA provide interpretable simplex-constrained representations of dermatoscopic images?

The journal asks:

> Are these archetypal coordinates reliable, reproducible, geometrically robust, and useful for direct model explanation?

The four investigations form a dependency chain rather than four unrelated projects:

1. **Stability** asks whether independently trained models recover the same semantic coordinate system after permutation alignment.
2. **Amortization gap** asks whether fast encoder coordinates are faithful to the frozen model's per-instance optimum.
3. **Controlled relaxation** tests whether bounded flexibility in the global simplex geometry improves fit without abandoning interpretability.
4. **Counterfactuals** use the validated coordinate system directly to explain what coordinate changes alter model predictions.

Stability validates the identity of the axes. Amortization analysis validates each sample's inferred location. Relaxation tests the robustness of the geometric assumption. Counterfactuals turn the resulting coordinates into model-behavior explanations. Together they support or delimit the reliability claim.

## Planned evidence progression

- Reproduce the exact conference baseline with traceable data and outputs.
- Build deterministic split manifests and multi-seed run infrastructure.
- Align run-specific archetypes on a validation/reference set and evaluate stability only on held-out test data.
- Freeze trained models and compare encoder-only inference with label-free local refinement at controlled step depths.
- Obtain approval for relaxation mathematics/regularization; then ablate `delta = {0, 0.1, 0.25, 0.5}` with `delta = 0` as the baseline recovery check.
- Implement simplex-coordinate counterfactual search only after the decoder/classifier interfaces and coordinate validity are trusted.
- Run final statistical repetitions, freeze claims, and write the new title/abstract/conclusion and explicit extension statement.

## Novel-material accounting

Maintain a manuscript ledger with each section, figure, and table marked `CONFERENCE`, `REUSED-WITH-CITATION`, `REWORKED`, or `NEW`. Candidate new journal material comprises:

- multi-seed stability protocol, matched stability metrics, uncertainty summaries, and visual comparisons;
- amortization-gap method, depth/runtime trade-off, and held-out measurements;
- relaxation formulation, regularization analysis, ablation, and stability interaction;
- simplex-coordinate counterfactual method, validity/proximity/plausibility evaluation, and medical-language limitations;
- integrated discussion of when archetypal coordinates are and are not trustworthy.

The 30% threshold must be verified against the publisher's actual counting convention; it is not guaranteed merely by listing four new experiments.

## Claim discipline

- Protocol documents contain planned analyses, not results.
- Report distributions and uncertainty across seeds, not best-run anecdotes.
- Separate validation used for alignment/tuning from held-out testing.
- Keep model-behavior explanation distinct from clinical explanation or causal biology.
- Negative results remain informative: instability, a large amortization gap, or no benefit from relaxation directly answers the reliability question.

