# Conference paper analysis

## Identity

- Title: *Probabilistic Variational Archetypal Learning on the Simplex for Interpretable Medical Image Analysis*.
- Venue: DATA 2026, paper 223.
- Authors: Ali Yassin, Narjes Soudin, Abbass Rammal, and Mageda A. A. Sharafeddin.
- Official public record: [DATA 2026 abstract](https://data.scitevents.org/Abstract.aspx?idEvent=pUOomaLGDs4%3D).

## Exact contribution established by available evidence

The conference work extended probabilistic variational Deep Archetypal Analysis from the JAFFE facial-expression setting to interpretable dermatoscopic lesion analysis on HAM10000. According to the official abstract, the adaptation contributed:

- an RGB convolutional architecture for medical images;
- a supervised lesion-classification objective;
- a TensorFlow 2/Keras implementation;
- a scalable stratified data pipeline;
- preservation of fixed simplex-constrained geometry;
- experiments for `K = {3, 5, 7, 9}`;
- evaluation of reconstruction fidelity, simplex adherence, coefficient sparsity, semantic alignment, and interpolation behavior;
- use of JAFFE as a controlled reference and HAM10000 as the seven-class medical-image domain.

The project brief further identifies archetype usage, decoded archetypes, nearest-real-image analysis, class/semantic enrichment, classification, coefficient entropy/sparsity, reconstruction, and broader interpretability analysis as conference material. These are all **old material** for journal novelty accounting.

## Conceptual baseline

The preserved baseline implements

```text
mu(x) = a Z_fixed
```

where `a` is a softmax-constrained per-image mixture and `Z_fixed` is fixed regular-simplex geometry. The encoder also learns clipped `logvar`, uses Gaussian reparameterization only during training, and decodes deterministic `t = mu` during evaluation. The loss is weighted reconstruction MSE, sparse classification cross-entropy, KL divergence, and a batch-derived simplex/archetype penalty. Exact page-level correspondence still requires the conference paper/source.

The final May baseline does **not** train with coefficient entropy stabilization. Normalized coefficient entropy and `1 - entropy` sparsity are evaluation metrics. January history contained entropy/temperature-related training logic, but it was removed before the matching May results and is not part of the working conference baseline.

## What the journal cannot claim as new

- Applying DeepAA to HAM10000.
- Fixed simplex-constrained coefficients by themselves.
- The original RGB/convolutional adaptation and supervised classification objective.
- The original `K` comparison.
- Baseline reconstruction, classification, entropy/sparsity, usage, decoded-archetype, nearest-image, enrichment, interpolation, or interpretability analyses.

Journal novelty must instead arise from the new reliability question and evidence: permutation-aware multi-run stability, amortization-gap measurement, controlled geometry relaxation, and direct simplex-coordinate counterfactual explanation.

## Extension-writing obligations

- Cite the conference version wherever prior methods/results are reused.
- Cite source captions for reused figures/tables.
- Avoid large verbatim passages.
- Use a new title, abstract, and conclusion.
- Include a clear conference-versus-journal differences statement.
- Maintain an explicit old/new mapping during drafting and demonstrate at least 30% new material under the publisher's rule.

## Gate 0 reproduction evidence

- Working baseline: `main` at `29acec505161530f9ea811534b05dd52d8df9899`, high confidence but not formal provenance certainty.
- All K=3/5/7/9 committed CSV-derived metrics recompute; only KL and the batch simplex penalty require checkpoint state.
- All four committed HDF5 checkpoints load under the documented reproduction environment and reproduce all eight checked metrics per K within `1e-5` (maximum observed difference below `2e-8`).
- The exact historical split was reconstructed and persisted: 8,013 train, 1,001 validation, and 1,001 test images.
- The conference split is image-ID disjoint but not lesion-ID grouped. There are 357 train/validation, 362 train/test, and 50 validation/test shared lesion IDs; 22 occur in all three and 725 unique lesions cross any boundary.

## Missing evidence

The conference PDF/source package, submitted-code archive, exact historical environment, complete original invocation logs, training histories, publisher correspondence, and author confirmation of the commit remain unavailable. Hyperparameters not persisted by artifacts are explicitly classified as inferred or unknown in `repro/configs/conference_runs.json`. This file is therefore not yet a page-level paper audit.
