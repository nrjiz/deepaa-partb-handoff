# JAFFE v1: proposed, not accepted

Question: under one fixed participant-disjoint allocation, do independent DeepAA
initializations learn repeatable coefficient representations of posed expressions
after validation-only permutation matching? Unseen participants are held out.
This is not emotion recognition ground truth or a clinical study.

| Component | Inherited rule / proposed JAFFE adaptation |
|---|---|
| Model | Original hash-verified DeepAA architecture and stochastic sampling; no Route B |
| Training | K5/latent4 and K7/latent6; seeds113/211/307/401/503; reference113; 50epochs; batch32; Adam0.001; unchanged losses; final checkpoint; fresh initialization |
| Data | **New** JAFFE v2, 213 TIFFs; separate manifests and output/claim namespace |
| Labels | **New** AN anger, DI disgust, FE fear, HA happiness, NE neutral, SA sadness, SU surprise; requested poses, not actual emotional state |
| Input | **New** 8-bit grayscale TIFF decode with Pillow, replicate into three channels, inherited TF bilinear128 resize/div255 float32; no crop, alignment, augmentation or histogram normalization |
| Split | **New proposal** six train/two validation/two test participants; fixed IDs in proposal.json; no image-level leakage; retain participant-by-class coverage |
| Matching | Inherited image-row validation coefficients, Pearson cost or2×MAE for SD≤1e-8, saved solver assignment and method masks; no test tie breaking |
| Test units | **New** paired image-row coefficient metrics; participant-specific image-metric means also reported. Never average different expression coefficients into a single person vector |
| Uncertainty | **New** paired participant-cluster resampling of within-person image-metric means (equal participant weights);1000 draws,seed2026,percentile95%; only2 test participants: discrete descriptive intervals, not reliable population inference |
| Semantics | **New** coefficient-weighted requested-expression distributions; seven ordered classes. Graded human ratings not used as training labels or this primary semantic metric |
| Other metrics | Inherited top10 latent nearest images with expression labels, matched SSIM/MSE, exact zero-mass handling, degenerate SD threshold, raw/matched distinction |
| Aggregation | Four reference comparisons per K, all seeds retained, descriptive mean/sampleSD/min/max; shared reference dependence; no training-seed CI |
| Validation | New JAFFE-specific exporter/replay validator using shared model and low-level metrics. All nulls/reasons retained; numerical replay tolerances documented in README |

All ten archive participants contain all seven expression labels. Counts and
nonuniform class frequencies are audited from actual filenames, then rechecked
against downloaded image bytes. The filename table is not proof of image-content
authenticity. Ratings rows do not exactly enumerate available images; the adapter
does not manufacture missing images or use ratings identifiers as membership.

## Decisions required before scientific execution

1. Accept this fixed 6/2/2 unseen-person design and its very weak participant
   uncertainty, or prospectively specify a different study (e.g. grouped folds).
2. Accept the full proposed adaptation: grayscale-to-RGB, hard requested-pose
   supervision rather than graded ratings, inherited architecture/loss/K/seeds,
   image metrics and participant-cluster estimand. There is no tuning on held-out
   results. The tiny training set may be unsuitable; completion is not usefulness.
3. After the partner audits their dataset/environment, approve exact config,
   manifest, clean source HEAD and individual attempts, and agree on preservation.

No HAM10000 scientific authorization is reused. Proposed decisions are not
activated by engineering tests. Model/hardware or protocol changes require a new
documented version and comparisons must not silently mix versions.

Official sources (checked2026-10-19): [dataset record](https://zenodo.org/records/14974867),
[format/labels/terms](https://zenodo.org/records/14974867/files/README_FIRST.txt?download=1),
[archive filename preview](https://zenodo.org/records/14974867/preview/jaffe.zip?include_deleted=0).
