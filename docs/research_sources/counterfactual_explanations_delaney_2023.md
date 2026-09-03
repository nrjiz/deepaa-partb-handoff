# Delaney et al. (2023): counterfactual explanations for misclassified images

## Source

Eoin Delaney, Arjun Pakrashi, Derek Greene, and Mark T. Keane, “Counterfactual Explanations for Misclassified Images: How Human and Machine Explanations Differ,” *Artificial Intelligence* 324 (2023), article 103995, DOI `10.1016/j.artint.2023.103995`. [Author manuscript / arXiv record](https://arxiv.org/abs/2212.08733).

## Relevant concept

Counterfactual image explanations contrast an observed model outcome with an input whose prediction changes. Image-space minimal edits are not automatically meaningful or psychologically adequate. Delaney et al. report that human-produced counterfactuals tend toward larger semantic changes and target-class prototypes rather than purely minimal pixel edits.

Latent or generative search is relevant because a decoder can constrain candidates toward a learned image manifold. A general prototype-guided formulation augments target-class success and proximity with attraction toward a target-class representation. The exact prototype-optimization equation must be attributed to its original method when used; Delaney et al. provide motivation and comparison, not blanket provenance for every such objective.

## What this project borrows conceptually

- Counterfactuals should be evaluated for meaningfulness and representativeness, not only minimal numerical change.
- Target-class prototype proximity is a candidate later objective/evaluation term.
- A generative representation can make image counterfactual search more structured than unconstrained pixel edits.

## What this project changes

The initial journal experiment searches directly over archetypal coefficients:

```text
a_cf in Delta^K
x_cf = decoder(a_cf Z)
```

with target-prediction loss plus `lambda * ||a_cf - a||_1`. This replaces an opaque latent edit with explicit changes in archetype membership. A target-class prototype in coefficient space is a later extension, not part of the initial protocol.

## Why simplex-coordinate search may be more interpretable

- Coordinates are nonnegative and sum to one.
- Changes can be reported per archetype, such as `A2: 0.61 -> 0.24`.
- L1 distance has a direct mixture-weight interpretation.
- The search remains inside the convex coordinate domain by construction when logits plus softmax are used.

These advantages depend on archetype stability and semantic validity; simplex membership alone does not guarantee a meaningful explanation.

## Limitations

- Decoder outputs may be blurry or off-manifold despite valid coefficients.
- A target-class prediction can be achieved through classifier artifacts.
- L1 closeness in coefficients need not imply perceptual or clinical closeness.
- Prototype attraction can trade sparsity for representativeness.
- Counterfactual methods can be unstable across initialization, target, or hyperparameters.
- The source studies general image explanation behavior, not dermatological validity.

## Medical interpretation warning

Generated images and coefficient paths explain the trained model's behavior. They do not show lesion progression, treatment response, biological mechanisms, actionable patient changes, or causal pathways. Clinical experts must review language and outputs before publication.

