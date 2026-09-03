# Archetype stability experiment v1

Status: **PROPOSED AND PRE-REGISTERED; NOT EXECUTED**.

## Scientific question

When DeepAA is retrained independently on the same leak-free HAM10000 split, does it recover consistent archetypal semantics after accounting for archetype permutation?

## Fixed primary scope

- Split: `PROPOSED JOURNAL SPLIT V1`, identical across every run and pending senior-developer acceptance.
- K values: `{5, 7}`; latent dimensions `{4, 6}` respectively.
- Training seeds: `{113, 211, 307, 401, 503}`.
- Total primary runs: 10.
- Predetermined reference: seed 113 independently within each K.
- The reference is not chosen by appearance, validation results, or test results.
- A future consensus analysis may be added only as a separately labeled analysis; it must not silently replace the reference rule.

## Frozen model and training configuration

The experiment retains the final May conference architecture and objective unchanged. It uses 128×128 RGB input, batch size 32, 50 epochs, Adam at learning rate 0.001, no early stopping, and final-epoch weights. Loss weights are reconstruction 1.0, classification 1.0, KL 0.01, and archetype/simplex 1.0. The encoder uses softmax coefficients, `mu = a @ Z_fixed`, learned log-variance, and training-time posterior sampling. There is no coefficient-entropy or temperature training objective.

Preprocessing remains JPEG RGB decode, bilinear resize to 128×128, float32 division by 255, and no augmentation. These values are explicitly frozen for this proposed experiment from source/config evidence; `repro/configs/conference_runs.json` retains the distinction between explicit, inferred, and unknown historical values.

The conference entry point cannot consume persisted manifests directly. Before execution, a reviewed launcher/data adapter must use the journal_v1 manifests while reusing the authoritative model and objective. That infrastructure must not alter the conference file or silently call its image-level splitter.

## Randomness and reproducibility

Each run must execute in a fresh process. Before importing/building stochastic training state, set `PYTHONHASHSEED` where applicable, then set Python `random.seed(seed)`, NumPy, and TensorFlow seeds. `tf.keras.utils.set_random_seed(seed)` is preferred as the combined TensorFlow/Keras control, with the final implementation documenting any additional `tf.random.set_seed` call.

Build the model and optimizer only after seeding. The training dataset shuffle must receive the run seed explicitly, retain `reshuffle_each_iteration=True`, and use deterministic data-pipeline ordering. Enable TensorFlow deterministic operations where technically supported and record whether this succeeds. Record package versions, device, deterministic-operation setting, thread settings, and all warnings.

Initialization, shuffle order, and posterior-sampling streams intentionally differ across different seeds. Repeating the same seed in the same environment/hardware should reproduce them as closely as TensorFlow permits. The split-generation seed remains fixed at 113 and is independent of the training seed; manifests must never be regenerated per run.

## Validation-only Hungarian alignment

For each K, extract validation coefficients for seed 113 and each candidate in the identical manifest order. For reference archetype `i` and candidate archetype `j`, compute

```text
cost(i,j) = 1 - corr(validation_A_reference[:, i], validation_A_candidate[:, j])
```

where `corr` is Pearson correlation. Apply Hungarian linear assignment to the full K×K cost matrix. Fixed-simplex vertex distance is prohibited as a semantic matching criterion because `Z_fixed` is geometrically identical across runs.

The threshold and fallback for constant or nearly inactive columns remain an **OPEN DESIGN DECISION**. No fallback may be invented during analysis. If an affected column produces unstable/undefined correlation, preserve the raw matrix, flag the comparison, and stop that alignment until the rule is approved.

For every run preserve:

- raw, unaligned archetype numbering and outputs;
- validation coefficient matrix and manifest order;
- full correlation and cost matrices;
- reference-to-candidate permutation;
- aligned derived outputs without overwriting raw outputs.

## Held-out test evaluation

The test set is not used to choose the permutation. After validation alignment, evaluate these families on journal_v1 test data:

1. **Coefficient stability.** Average coefficients over images sharing each test `lesion_id`, then calculate matched-column Pearson correlation, coefficient MAE, full coefficient-vector L1 distance, and dominant-archetype agreement across the same 748 lesions. Image-level versions are secondary diagnostics.
2. **Semantic stability.** For each archetype, compute coefficient-weighted `P(dx | archetype)` over test images and compare matched seven-class distributions using base-2 Jensen–Shannon divergence. Diagnosis labels are evaluation-only, not alignment inputs.
3. **Nearest-real-image stability.** Within the test set, rank images by Euclidean distance from `mu` to each run's corresponding raw `Z_fixed` vertex. After permutation, compare matched top-10 sets by Jaccard overlap and compare their diagnosis distributions.
4. **Visual stability.** Deterministically decode each run's raw fixed vertices and compare the matched images. The exact image-similarity metric remains open and must be approved before results are computed.

Report each seed rather than selecting favorable runs. Analyze K=5 and K=7 separately. Preserve reference-versus-candidate results for all four candidates per K and distinguish primary lesion-level metrics from secondary image-level diagnostics.

## Open decisions before execution

- Definition of “constant or nearly inactive,” including numerical threshold and approved alignment fallback.
- Exact decoded-image similarity metric and whether any qualitative review is included.
- Uncertainty/interval method for aggregating the four reference comparisons.
- Final run-output schema and implementation review for the manifest-aware data adapter, deterministic controls, alignment tests, and overwrite protection.

Until these decisions and the proposed split are reviewed, Gate 1 is prepared but not ready to launch training.
