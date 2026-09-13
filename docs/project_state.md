# Project state

Last audited: **2026-09-13** (second-run validation and first stability pair).

## First held-out stability comparison: 2026-09-13

Seed211 completed all 50 epochs at execution revision `f097f2f9e6ad72464685dacca59ba95ba273ab33`, recorded clean before launch. Artifact validation is **PASS WITH LIMITATIONS**: 22 inventory entries match checksums/sizes (24 total run files), all epochs finite, manifests/source unchanged, mappings in frozen order, finite simplex coefficients/latents, original checkpoint reload exact and independent reload reproduces decoded archetypes exactly. Recorded elapsed interval is 62,550.249728 s (17h22m30.250s), not a claim of uninterrupted active CPU time. Seed113's previous validation was reused after rechecking all original hashes.

Existing validation-only Hungarian and test tools were executed without code or methodology changes. D-002 already became SUPERSEDED in `7911e864...`; accepted D-015/D-016 govern matching. Candidate raw archetypes `[4,3,5,2,1]` align to reference `[1,2,3,4,5]` (zero-based permutation `[3,2,4,1,0]`); all 25 cells use Pearson, zero fallback cells. Test analysis uses this fixed permutation, not a test-selected or visual match.

For 748 test lesions, coefficient-vector L1 mean is **0.347920** (paired lesion bootstrap95% interval **0.336131–0.358744**); dominant agreement **0.506684** (**0.469251–0.541444**). Matched lesion Pearson values are **0.701773, 0.787872, 0.490117, 0.567261, 0.050737**, all defined. Mean image-coefficient-weighted semantic JS(base2) is **0.001615**; nearest-image and decoded-image metrics are retained per archetype in the registry/report. These are one pair's numerical findings, not overall reproducibility, clinical validity or a completed multi-seed study. Weak A5 correspondence was retained without tuning/rematching.

Derived root: `research/analysis/stability_v1/K5_113_vs_211_20260913/` (alignment, test, presentation). Labeled side-by-side figure: `presentation/decoded_archetypes_aligned.svg`, embedding original PNG bytes. Separate validation/report: `C:/Users/UserPc/Desktop/ai/stability_pair_20260913/`. Original runs/logs are unchanged. Two of ten runs and one of eight planned reference/candidate comparisons are complete; remaining runs/comparisons require separate authorization. Historical readiness/no-results statements below describe earlier audit dates and are superseded by this dated update.

## Second-run preparation: 2026-09-09

K5/latent4/seed211 remains **PLANNED, NOT STARTED**. Preparation began from clean HEAD `42ec30f84cf826a2a3465fbb0778af4b04fc0bd5`; there were no pre-existing changes. Research-runner instrumentation now adds flushed phase/epoch/batch telemetry and incremental atomic epoch history without changing the protected model, objective, seeds, data ordering or final checkpoint rule. New `progress.jsonl` is closed before inventory hashing; existing final history columns are retained. This preserves progress records, **not resumable training**.

Bounded old/new loop comparison used fresh seed211 processes, 65 training images and 10 evaluation images drawn solely from frozen training membership, two miniature epochs and six optimizer steps per path. Losses, batch sequence, final weights, optimizer state and history matched exactly (maximum difference 0); epoch-1 history was present before epoch 2. Evidence: `C:/Users/UserPc/Desktop/ai/seed211_preparation_20260909_b6e42f/equivalence/comparison.json`. This is engineering evidence, not a primary run or full-trajectory guarantee. See `research/compute_readiness.md` for checks, limitations, recovery assessment and proposed manual command.

Reference seed113 and its external log match the previous validation hashes; manifests/source unchanged. All 10,015 manifest image paths exist in the two documented archive folders. Intended seed211 output/log are unused. RAM availability was low and variable during preparation, so no launch-readiness guarantee follows from this snapshot. User review/commit, final clean-revision preflight, fresh resource/process checks and separate manual launch are required. No comparison or primary training was executed.

## Current execution update: 2026-09-09

The first primary reference run, `research/runs/stability_v1/K5/seed_113`, completed all 50 epochs at execution revision `0e32c8d469be141d30e627219aeac7ee253a34fa`. Validation verdict: **PASS WITH LIMITATIONS**. All 21 inventoried artifacts match recorded hashes/sizes; protected source and frozen manifests match; independent checkpoint reload succeeds with 30 finite tensors and exact decoded-archetype reproduction. Validation/test mappings match manifest order, coefficient/latent arrays pass finite/simplex checks, and nearest-image/class tables reproduce from saved arrays. No required single-run artifact is missing.

The unchanged baseline actually optimizes batch-mean per-image **summed** reconstruction error, not pixel-mean MSE. The runner reuses this objective with weights `1/1/0.01/1`; no loss-objective mismatch was found. Final-checkpoint train/validation/test pixel MSE is `0.005987259559 / 0.006343114190 / 0.006468955427`; accuracy is `0.681063538 / 0.682634731 / 0.678642715`. Loss components are unweighted batch averages, whereas pixel MSE and accuracy are image-weighted. Full metrics and the normalization audit are in the registry and separate report. `scientific_evidence: null` is the writer's primary-run default, not a scientific verdict.

Recorded runner elapsed time is **14h 27m 16.698s**, including initialization, training, evaluation and exports; phase/active-compute durations are unavailable. The earlier 3-warm-up/20-timed-step CPU benchmark (median 1.517 s, range 1.309–1.811 s) extrapolated **5.29 training-only hours**, not total elapsed time. One completed run establishes local execution feasibility but does not establish matrix practicality or multi-seed stability. Assess progress/phase logging and interruption recovery before another run; no implementation or training was authorized in this validation task.

Separate validation: `C:/Users/UserPc/Desktop/ai/validation_K5_seed113_20260909/REPORT.md` and `checks.json`. External log: `C:/Users/UserPc/Desktop/ai/primary_K5_seed113_0e32c8d_20260905_a7f19c.log`. Benchmark: `C:/Users/UserPc/Desktop/ai/cpu_feasibility_20260905/RESULTS.md`. Original artifacts/log remain unchanged. See `docs/experiments/experiment_registry.md` and `research/compute_readiness.md` for the dated outcome. Older readiness/no-primary-output statements below describe their original audit dates and are superseded by this update, not deleted historical evidence.

Compute/transfer follow-up **2026-09-05**: current research branch is `journal-research-local` at `7911e8643495a2f34820ca0adedbea1f322ab8da`; local `main` and locally cached `origin/main` remain the conference baseline `29acec5`, and origin push remains `DISABLED` (no live remote contact). The worktree was clean before this documentation update. No primary outputs exist. Hardware inventory identifies Intel UHD Graphics 620; TensorFlow sees no GPU, and WSL is not installed. No accessible external GPU target has been established. The bundle is valid, but an LF checkout fails the source guard's CRLF-based hash. Thus the earlier Gate 1 readiness statement describes local infrastructure validation, not current external launch readiness. See `research/compute_readiness.md` and `research/external_gpu_execution.md` for the audit and required decisions.

## ESTABLISHED CONFERENCE FACTS

- Conference paper: *Probabilistic Variational Archetypal Learning on the Simplex for Interpretable Medical Image Analysis*, DATA 2026, paper 223, by Ali Yassin, Narjes Soudin, Abbass Rammal, and Mageda A. A. Sharafeddin.
- The official abstract describes a probabilistic variational DeepAA adaptation implemented in TensorFlow 2/Keras with an RGB convolutional architecture, supervised lesion-classification objective, scalable stratified pipeline, and fixed simplex-constrained geometry.
- HAM10000 is the primary medical-image dataset and JAFFE is a controlled reference domain.
- The reported study considered `K = {3, 5, 7, 9}` and evaluated reconstruction fidelity, simplex adherence, coefficient sparsity, semantic alignment, and interpolation behavior.
- The supplied project brief additionally identifies classification, archetype usage, decoded archetypes, nearest-real images, class/semantic enrichment, and interpretability analysis as conference material.
- Conference material is prior work. It must be cited and must not be presented as a new journal contribution.

The conference PDF/source package is not present locally. The authoritative GitHub repository and its committed result artifacts are now available, but the paper is still needed for a figure-by-figure and table-by-table reuse inventory.

## AUTHORITATIVE REPOSITORY PROVENANCE

- Remote: `https://github.com/Amy007-ali/Deep-Archetypal-Analysis-on-HAM10000.git`.
- Local clone: `C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000` (workspace fallback because the elevated clone process could not access the preferred `Documents\github` location).
- Branch: `main`; only `main`, `origin/main`, and the `origin/HEAD -> origin/main` symbolic reference are present.
- Working conference baseline revision: `29acec505161530f9ea811534b05dd52d8df9899`, authored and committed `2026-05-09T15:15:47+03:00`, subject `adding results for K=9`.
- Project HEAD at the start of Gate 1 infrastructure implementation: `feafd3ac0a0837a5d58ebd3c121064a51f8e03d3`, subject `research: freeze lesion-grouped split and stability protocol`. This later project commit does not redefine the preserved conference baseline.
- Tags: none.
- History: seven commits from 2026-01-10 through 2026-05-09. The May sequence adds result bundles for K=5 (`661cc7f`), K=3 and K=7 (`6dd8154`), then K=9 (`29acec5`).
- The model source and README are byte-identical from `661cc7f` through HEAD; the last two commits add results only.

**Baseline assessment:** the senior developer has designated `29acec5` as the working conference baseline. Confidence is high because it contains the TensorFlow/Keras variational source, all four published-result bundles, and checkpoints that reproduce every checked committed metric. Provenance is not formally certain because there is no conference tag/release, submitted archive, exact invocation log, or author confirmation.

Committed artifact matches:

| K | Run directory | MSE | KL | Simplex penalty | Accuracy | Entropy | Sparsity | Mean Var(a) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | `20260508_002312` | 0.008679 | 9.052050 | 0.333270 | 0.666334 | 0.897674 | 0.102326 | 0.022459 |
| 5 | `20260507_131506` | 0.006598 | 18.027552 | 0.558307 | 0.672328 | 0.913765 | 0.086235 | 0.010752 |
| 7 | `20260508_104147` | 0.004913 | 27.029241 | 0.653163 | 0.695305 | 0.912072 | 0.087928 | 0.006778 |
| 9 | `20260509_112437` | 0.004175 | 36.024504 | 0.710956 | 0.709291 | 0.916249 | 0.083751 | 0.004350 |

## GATE 0 REPRODUCTION STATUS

Gate 0 is **PASS WITH DOCUMENTED LIMITATIONS** as of 2026-09-03.

- Repository-local `docs/` is now the canonical documentation location. The original external `C:\Users\UserPc\Desktop\ai\docs` copy is preserved.
- `repro/baseline_artifact_hashes.csv` inventories all 226 committed May result files (99,940,603 bytes), including the four checkpoint hashes. Source and README hashes are frozen in `repro/baseline_inventory.md`.
- The isolated evaluation environment is `C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro`; its complete current package freeze is recorded. This is not asserted to be the unknown historical environment.
- The exact historical image-level split is persisted at `repro/splits/conference_image_level`: 8,013 train, 1,001 validation, and 1,001 test images, with stable manifest hashes.
- The split is image-ID disjoint but lesion-ID overlapping: 357 train/validation, 362 train/test, 50 validation/test, 22 in all three, and 725 unique lesion IDs crossing any boundary. Affected images total 1,737 (train 932, validation 400, test 405).
- Artifact-only verification found 192 exact numeric matches and 24 usage matches at the committed one-decimal precision. KL and simplex penalty require checkpoint state because the CSVs omit `logvar` and `z_pred`.
- All four HDF5 checkpoints loaded into the unmodified source. Evaluation-only reproduction yielded 32/32 metric matches at absolute tolerance `1e-5`; maximum observed difference was below `2e-8`.
- The baseline invariant script passed 30/30 checks without training.
- The final May baseline has no coefficient entropy/temperature training objective. Entropy and `1 - entropy` sparsity are post-training evaluations only.

## GATE 1 INFRASTRUCTURE STATUS

Gate 1 is **READY FOR PRIMARY TRAINING** as of 2026-09-03. The suite passed 27/27 tests, the non-scientific smoke passed, and the matrix dry-run emitted the exact ten commands without execution. No primary run has been executed.

- All 10,015 metadata rows have unique `image_id`, non-missing `lesion_id`, and diagnosis labels consistent within lesions; zero lesion IDs have multiple `dx` values.
- Accepted immutable `journal_v1` was generated once using `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=113)`, grouped by `lesion_id` and stratified by `dx`. Fold 0 is test, fold 1 validation, and folds 2–9 train; no alternative seed/fold search was performed. A future replacement must be a new `journal_v2`, never a mutation.
- Counts are 8,011/1,002/1,002 images and 5,975/747/748 lesions for train/validation/test. All seven diagnoses occur in each partition, every image occurs exactly once, and every pairwise lesion overlap is zero. All 17 split invariants passed.
- The historical split remains unchanged under `repro/`; the accepted journal split and comparison are under `research/splits/journal_v1/`.
- `archetype_stability_v1` is accepted and pre-registered but not executed: K=`{5,7}`, seeds=`{113,211,307,401,503}`, ten planned runs, seed 113 reference within each K, validation-only Hungarian matching, and test-only stability evaluation.
- The planned architecture/objective remains the final May baseline. Planned explicit values are 128×128 RGB, batch 32, 50 epochs, Adam at 0.001, final-epoch checkpoint, and loss weights `1/1/0.01/1`; coefficient entropy remains evaluation-only.
- Degenerate validation columns use population standard deviation `<= 1e-8` and deterministic `2 * MAE` alignment cost. Visual comparison uses RGB SSIM at data range 1.0, with pixel MSE secondary. Four seed comparisons are summarized without inferential confidence intervals; specified lesion metrics use the fixed paired bootstrap.
- The manifest-only data adapter, deterministic runner, authoritative-model adapter, alignment, metrics, output protection, tests, smoke check, and external-GPU guide are implemented under `research/`. The smoke run is explicitly non-scientific.
- Local TensorFlow 2.21 sees no GPU. The machine has 8 logical CPU threads and low disk headroom, making ten full local CPU runs impractical. The external route is documented in `research/external_gpu_execution.md`.

## CURRENT JOURNAL PROPOSALS

These are planned investigations, not findings:

1. Multi-seed, permutation-invariant archetype stability.
2. Measurement of the encoder amortization gap by label-free local test-instance refinement.
3. Bounded radial relaxation of the fixed simplex.
4. Counterfactual optimization directly in simplex coordinates.

## ACCEPTED DECISIONS

- One main journal problem connects all four investigations: reliability and robustness of interpretable simplex coordinates with fast inference.
- Stability alignment must account for permutation of archetype identities.
- Held-out diagnosis labels must not be optimization signals in amortization-gap measurement.
- Counterfactuals explain model behavior and must not be described as biological progression, treatment advice, or clinical causality.
- `delta = 0` must reproduce the approved conference baseline once that baseline is identified.
- Gate 3 mathematical and regularizer design approval is required before simplex-relaxation code is written.
- `journal_v1` is the immutable split for stability v1; replacements require a new `journal_v2`.
- Stability v1 uses K=`{5,7}`, the five fixed seeds, seed 113 references, validation-only matching, and test-only evaluation.
- Degenerate alignment columns use the registered `2 * MAE` fallback; visual comparison and lesion bootstrap specifications are frozen in D-016 through D-018.

## OPEN DECISIONS

- Seek author/coauthor confirmation that working baseline HEAD `29acec505161530f9ea811534b05dd52d8df9899` is the submitted conference snapshot, or preserve any contrary evidence if found.
- Recover exact commands/configurations for the four May runs if archival records exist; currently inferred fields and their evidence are explicit in `repro/configs/conference_runs.json`.
- Senior-review and version the Gate 1 implementation so a primary runner can satisfy its clean-worktree guard.
- Provision sufficient GPU and persistent storage before launching the ten primary runs.
- Select the local-refinement objective, parameters, optimizer, learning rate, constraints, stopping rule, and runtime protocol after the authoritative model is available.
- Resolve whether simplex relaxation changes the target of the archetype regularizer from `Z_fixed` to `Z_delta`, changes the regularizer itself, or uses another approved formulation.
- Decide whether radial scale parameters need an identifiability/normalization constraint beyond the interval bound.
- Specify the classifier used in counterfactual search, target selection policy, `lambda`, success threshold, plausibility checks, and evaluation metrics.
- Determine the exact bibliographic source intended by “prototype-based objective”; Delaney et al. (2023) motivates prototype-like human edits but is not itself the source of every prototype-guided optimization equation.

## BLOCKERS

- Repository-local `docs/` is canonical; the original external docs copy is intentionally retained.
- Conference tag/release, paper source/PDF, submitted-code archive, and explicit author confirmation of the baseline commit are absent.
- Exact historical run commands, environment lock, and training histories are absent. Gate 0 now provides a current reproduction freeze, saved historical split manifests, evidence-classified run configuration, and no-training invariant checks under `repro/`; these are new preservation records, not recovered historical files.
- The historical source uses per-image class-stratified splitting rather than grouping by `lesion_id`; repeated lesions cross partitions. The historical membership and quantified overlap are persisted under `repro/splits/conference_image_level`; journal runners bypass that splitter and consume `journal_v1` manifests.
- Historical training shuffling does not pass an explicit seed to `tf.data.Dataset.shuffle`. Journal runners add an explicit run seed and deterministic pipeline controls without modifying the historical source; cross-hardware bitwise identity is still not guaranteed by TensorFlow.
- No code-level early stopping/model-selection rule is present; training runs for the requested epoch count and saves final weights.
- The source contains no coefficient-entropy stabilization term at HEAD. Entropy and `1 - entropy` sparsity are post-training evaluation metrics. The January source did contain temperature scaling and an entropy term, both removed in commit `661cc7f` before the matching May results.
- The earlier local HAM implementation at `C:\Users\UserPc\Documents\github\ham10000_aanet` is a materially different deterministic PyTorch project, not the conference implementation.
- The earlier PyTorch encoder emits unconstrained Euclidean coordinates. Its barycentric coordinates may be negative during inference; the loss only penalizes points outside the simplex, and metrics later clamp/renormalize coefficients. It must not be described as a hard per-sample simplex constraint without qualification.
- The earlier PyTorch script's logistic regression is fitted and scored on the same held-out test coefficient matrix. Its reported downstream accuracy/AUC/F1 therefore cannot be used as an unbiased journal classification result.
- The earlier PyTorch split is image-level, is not saved as a manifest, and can place images sharing one `lesion_id` into different partitions.
- No checkpoints, `aanet_outputs` directory, configs, tests, dependency file, or documentation were found alongside the earlier PyTorch HAM script.
- The conference manuscript/source, publisher invitation text, and source-paper files were not present locally. Their exact wording and reuse inventory still require author-provided copies.

## DATASET STATUS

HAM10000 is already present and complete in two local locations:

- Script-default location: `C:\Users\UserPc\Downloads\archive\`
  - `HAM10000_metadata.csv`
  - `HAM10000_images_part_1\` — 5,000 JPEG files
  - `HAM10000_images_part_2\` — 5,015 JPEG files
- Duplicate working copy: `C:\Users\UserPc\Desktop\HAM10000\` with the same filenames and counts.

The metadata has 10,015 rows, 10,015 unique image IDs, 7,470 unique lesion IDs, and columns `lesion_id,image_id,dx,dx_type,age,sex,localization`. The two local metadata copies have the same SHA-256 hash: `06AEE5E603D313BA69DE38263053C792A1DFF030EB9C44A96CEB477CEEFB99A5`.

The Desktop extracted dataset occupies 2,903,469,944 bytes (about 2.704 GiB). `C:\Users\UserPc\Desktop\HAM10000.zip` adds 2,810,020,913 bytes (about 2.617 GiB). Keeping one extracted copy plus one archive needs roughly 5.3 GiB; the current duplicate extracted copies and archive use substantially more.

The earlier PyTorch script contains preprocessing but no downloader:

- Metadata shuffle with `random_state=seed`.
- Optional diagnosis include/exclude filter and sample limit.
- Lookup by `image_id` across two image directories.
- PIL RGB conversion, direct resize to square (default 64×64), float32 conversion, and `/255` scaling.
- No augmentation, aspect-ratio preservation, normalization beyond `/255`, lesion-group split, persisted split manifest, or explicit corrupt/missing-file report.
- Two seeded, stratified `train_test_split` calls produce train/validation/test partitions. This is repeatable for the same local file set and software behavior, but full training is not guaranteed bitwise deterministic because deterministic backend settings are absent.

No download is needed now. If reacquisition is needed later, the official ISIC 2018 Task 3 page provides direct downloads under CC BY-NC and states the required attribution; this route does not inherently require Kaggle. A later direct-download automation is technically feasible after explicit authorization and a human license/attribution review. A Kaggle mirror would require a Kaggle account/API token and any displayed terms. Dataset download remains prohibited until explicitly authorized.

## Repository/code inventory

### Authoritative conference HAM10000 implementation

- `C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\daa_ham10000_interpretability_fixed_v2.py`: sole training/model/evaluation entry point at HEAD.
- `DeepAA(tf.keras.Model)` contains a convolutional encoder, softmax coefficients `a`, `mu = a @ Z_fixed`, learned `logvar`, Gaussian reparameterization during training, image decoder, and lesion-classification head.
- The loss is `recon_w*MSE + class_w*CE + kl_w*KL + at_w*simplex`, where the simplex/archetype term compares batch-derived `z_pred = b @ mu` with `Z_fixed`.
- `latent_dim = K - 1`; README commands cover `K = {3,5,7,9}`.
- Dataset code auto-discovers HAM paths or accepts explicit paths, resizes RGB images to 128×128, scales to `[0,1]`, and performs seeded per-class 80/10/10 splitting by image.
- Result bundles contain four HDF5 weight checkpoints, per-test-image coefficients/interpretations, latent CSVs, generated archetypes, nearest-real examples, class/archetype tables, and other figures. No notebooks or separate configs are present.

### Earlier deterministic PyTorch HAM10000 implementation

- `C:\Users\UserPc\Documents\github\ham10000_aanet\aanet_ham10000.py`: only standalone training/evaluation entry point; defines `HAMDataset`, simplex utilities, `Encoder`, `Decoder`, `AANet`, training, metrics, and figures.
- `C:\Users\UserPc\Documents\github\ham10000_aanet\aanet_ham10000.ipynb`: Colab launcher/orchestration notebook; 8 Markdown and 10 code cells, primarily Drive mounting and invocations of the Python script.
- Default outputs: `./aanet_outputs/<run-tag>/best_model.pt`, `history.csv`, `metrics.csv`, `alpha.csv`, `class_dist.csv`, `curves.png`, `archetypes.png`, `nearest_real.png`, and `histogram.png`. No such output directory was found in the audited locations.

### JAFFE code still present outside the workspace

- `C:\Users\UserPc\Desktop\jaffe_project\main.py`: legacy monolithic dataset/model/training script defining `JAFFEDataset` and a simple `DeepAA`; no split or evaluation.
- `C:\Users\UserPc\Desktop\jaffe_project\visualize_archetypes.py`: duplicate model definition and visualization.
- `C:\Users\UserPc\Desktop\jaffe_project\resize_images.py`: offline 128×128 resizing.
- `C:\Users\UserPc\Desktop\jaffe_project\deep_aa_model.pth`: existing legacy checkpoint (1,241,594 bytes).
- `C:\Users\UserPc\Desktop\jaffe_colab_deepaa\`: more faithful PyTorch replication with `faithful_jaffe_deepaa/data.py`, `model.py`, `plotting.py`, `utils.py`, and `train_jaffe_deepaa_faithful.py`; root launcher duplicate; README and `requirements.txt`; no run outputs found.
- `DeepAAJaffe` in `model.py` has a fixed simplex buffer, variational `mu`/`sigma`, image decoder, side-information decoder, and reconstruction/side/archetype/KL losses. It is a useful reference, not automatically the conference HAM implementation.

### Tests, docs, and environments

- The historical authoritative repository had no dependency/environment file. Gate 0 adds a clearly labeled current reproduction freeze under `repro/`; historical package versions remain unknown.
- The historical authoritative repository had no automated tests. Gate 0 adds a standalone 30-check baseline invariant script under `repro/tools/`; it requires no retraining.
- The earlier PyTorch HAM implementation has no README or dependency/environment file. Imports imply Python, NumPy, pandas, Pillow, tqdm, PyTorch, scikit-learn, and Matplotlib.
- The faithful JAFFE directory has a five-package `requirements.txt` but does not list scikit-learn because it is not used there.
- The repository-local documentation knowledge base is now preserved under `docs/`; the external source copy is retained separately.

## NEXT ACTION

Senior developer/BRAIN should review the first K5 pair's validation, alignment and held-out findings, then separately authorize remaining frozen runs/comparisons without tuning from test results. Progress logging is implemented; resume is not, and interruption can lose model/optimizer progress despite incremental history. The senior developer handles Git and must restore a reviewed clean worktree before any guarded training launch. Preserve `journal_v1`, both completed runs and separate analysis evidence. This single comparison does not settle overall multi-seed stability.
