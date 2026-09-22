# Experiment registry

## D-020 attempt ledger preparation — 2026-09-21

- K5/latent4/seed503 original: `research/runs/stability_v1/K5/seed_503`, execution
  `4a5daa34f68d9a68b3b024bb87c7c9ff5c85ccc2`, interrupted without a resumable
  checkpoint (8 complete epochs; final recorded epoch9/batch150). Retained,
  never relabeled COMPLETED or excluded from the attempt audit.
- Fresh K5/seed503 GPU attempt: authorized operational design, **NOT STARTED**.
  Exact timestamped sibling, new execution revision and log are registered only
  at the senior's future reviewed Colab launch. Starts epoch1; no model resume.
- Remaining K7 seeds113/211/307/401/503: **NOT STARTED**, sequentially queued after
  the new seed503 attempt, with unchanged frozen settings and seed113 reference.
- Colab target GPU, environment, data-byte verification and timings: **PENDING**.
  No CPU/GPU numerical equivalence is claimed. Existing K5 work is validated
  before reuse; D-019 remains specific to seed401. See D-020 and the migration
  runbook for preservation, explicit path mapping, durable receipts and gates.

Two primary runs and one reference/candidate stability comparison are completed and validated as of 2026-09-13. Copy the template below once per run. Do not overwrite completed entries; update `status` and append dated notes.

## Run template

```yaml
experiment_id: ""
date: "YYYY-MM-DD"
git_commit_hash_manual: ""  # entered manually; never inferred from uncommitted changes
model_variant: ""
dataset: "HAM10000"
dataset_split_id: ""
K: null
random_seed: null
delta: null
refinement_steps: 0
important_hyperparameters:
  image_size: null
  batch_size: null
  epochs: null
  optimizer: ""
  learning_rate: null
  loss_weights: {}
  early_stopping: ""
checkpoint_path: ""
output_paths:
  run_directory: ""
  raw_metrics: ""
  coefficients: ""
  figures: ""
metrics: {}
notes:
  purpose: ""
  deviations: ""
  failures_or_warnings: ""
status: "PLANNED"  # PLANNED | RUNNING | COMPLETED | FAILED | EXCLUDED
```

## Registry entries

Add newest entries below this heading. A `COMPLETED` status means artifacts exist and checks passed; it does not mean a hypothesis was supported.

### 2026-09-13: seed211 validation and K5 seed113-versus-seed211 comparison

Seed211 status: **COMPLETED; validation PASS WITH LIMITATIONS**. Recorded execution revision `f097f2f9e6ad72464685dacca59ba95ba273ab33`, pre-run status empty; journal_v1, K5/latent4/seed211, full 8,011/1,002/1,002 partitions, RGB128 float32, batch32, 50 epochs, Adam0.001, weights1/1/0.01/1, no entropy/temperature objective, no early stopping, final epoch. Unchanged source/objective imported through the existing adapter. Run: `research/runs/stability_v1/K5/seed_211`; checkpoint `final.weights.h5`, SHA-256 `eaf2938382f6b2261a7870cd1a67bd6e6389a93f91876a94841c3e2766f278ed`. Original reload reports loaded/passed true, 30 tensors, maximum difference0; fresh read-only reload reproduces saved decoded arrays exactly.

All22 inventory hashes/sizes match;24 total files including inventory and completion. Epoch sequence1–50 and losses finite; progress confirms50 epoch ends and recorded batch milestones through251. Validation/test mappings match frozen row order; coefficient shapes1002x5, latent shapes1002x4, finite/nonnegative coefficients, maximum row-sum errors1.7881393e-7/2.3841858e-7 (<1e-5). Five decoded PNGs,50 nearest rows and35 class rows validated. No required artifact missing. Runtime UTC2026-09-09T12:45:27.945771+00:00 to2026-09-10T06:07:58.195499+00:00 =62,550.249728s; no inference of uninterrupted compute. `scientific_evidence:null` remains the primary writer default, not a validity verdict. Seed113 hashes match its prior validation; neither run/log was changed. Validation does not replay training or independently reevaluate all raw-image metrics.

Saved final seed211 checkpoint train/validation/test metrics (loss scalars are equal-weight batch averages; pixel MSE and accuracy are image-weighted):

| Metric | Train | Validation | Test |
|---|---:|---:|---:|
| Total | 287.308911 | 313.885802 | 315.510196 |
| Reconstruction sum | 285.571506 | 312.120415 | 313.751657 |
| CE | 0.882847 | 0.917459 | 0.925103 |
| KL before0.01 weight | 18.033367 | 18.032235 | 18.033089 |
| Simplex | 0.674225 | 0.667602 | 0.653107 |
| Pixel MSE | 0.005802981 | 0.006347880 | 0.006363141 |
| Accuracy | 0.680315 | 0.684631 | 0.684631 |

Analysis implementation revision: `f097f2f9e6ad72464685dacca59ba95ba273ab33`, initially clean. Existing `align_validation_coefficients.py` and `evaluate_stability_pair.py` ran without modification after readiness checks. Existing decision history shows D-002 changed from PROPOSED to SUPERSEDED and D-015/D-016 to ACCEPTED in `7911e8643495a2f34820ca0adedbea1f322ab8da`; no new historical approval was invented. Older “not executed” status text was stale, not a conflicting matching rule.

Alignment fitted on identical validation image order only: zero-based candidate columns `[3,2,4,1,0]` in reference order, equivalently reference raw1–5 maps to candidate raw4,3,5,2,1. All25 costs use `1-Pearson`; zero cells trigger population-std<=1e-8 fallback. Cost/method matrices, explicit fallback mask, assignment and permutation direction are saved. That same permutation was applied unchanged to test; no fixed-vertex distance or test/visual optimization selected correspondence.

Held-out metrics by reference archetype (coefficients averaged within748 lesions; semantic distributions coefficient-weighted over1002 images; nearest sets contain10 test images; visual metrics compare decoded RGB128 images at range1):

| Ref / candidate raw | Lesion Pearson | Lesion MAE | Semantic JS2 | Top10 Jaccard | Top10 dx JS2 | RGB SSIM | Pixel MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 / A4 | 0.701773 | 0.066457 | 0.002072 | 0.111111 | 0.052123 | 0.658692 | 0.065521 |
| A2 / A3 | 0.787872 | 0.053120 | 0.001160 | 0.176471 | 0.000000 | 0.801883 | 0.023640 |
| A3 / A5 | 0.490117 | 0.084139 | 0.001144 | 0.052632 | 0.051899 | 0.699238 | 0.091902 |
| A4 / A2 | 0.567261 | 0.059737 | 0.002206 | 0.052632 | 0.162256 | 0.597273 | 0.097906 |
| A5 / A1 | 0.050737 | 0.084467 | 0.001490 | 0.000000 | 0.311278 | 0.647862 | 0.089973 |

Lesion coefficient-vector L1 mean0.347919627 (95% interval0.336131114–0.358744425); dominant agreement0.506684492 (0.469251337–0.541443850). Only these two quantities received the frozen paired lesion bootstrap:1,000 resamples, seed2026, percentile2.5/97.5. Mean semantic JS2=0.001614554. All five test correlations are defined; the tool preserves undefined correlations as null/false rather than substituting values. No bootstrap was added for semantic, nearest, visual or per-column Pearson metrics.

Interpretation, distinct from findings: correspondence varies across archetypes; low A5 correlation and limited top10 overlap remain reported. Small diagnosis-distribution divergence does not establish interchangeable archetypes or clinical validity. One pair cannot establish overall reproducibility; four candidate comparisons per K are still required before the registered cross-seed summaries (no seed-level inferential CI).

Derived root `research/analysis/stability_v1/K5_113_vs_211_20260913/`: `alignment/` contains original tool cost/method matrices and aligned coefficients/permutation; `test/` contains stability summary and coefficient/semantic/nearest/visual tables; `presentation/` contains explicit fallback mask, assignment/direction, concise metrics and `decoded_archetypes_aligned.svg` with unmodified embedded PNGs. These are ignored local analysis artifacts, not automatically included in Git commits; preserve them separately. External validation/report root: `C:/Users/UserPc/Desktop/ai/stability_pair_20260913/` (`checks.json`, `readiness.json`, scripts, `REPORT.md`). No original run artifacts, existing analyses, or prior reports overwritten. No training or manuscript writing. Next: senior review this one pair and separately authorize remaining frozen runs; no retuning/rematching from these test results.

### 2026-09-09 preparation: stability_v1_K5_seed211

Status remains **PLANNED / NOT STARTED**. Intended output: `research/runs/stability_v1/K5/seed_211`; intended external log: `C:/Users/UserPc/Desktop/ai/primary_K5_seed211_20260909_b6e42f.log` (both absent at preparation). Execution revision is **pending user review/commit**, not the preparation HEAD `42ec30f84cf826a2a3465fbb0778af4b04fc0bd5` or seed113's execution revision. No metric-driven tuning: K5/latent4/seed211, journal_v1, RGB128 float32, batch32, 50 epochs, Adam0.001, reconstruction1 + CE1 + KL0.01 + simplex1; no early stopping, entropy training term or temperature scaling; final epoch checkpoint and existing deterministic controls.

Research-only instrumentation records flushed progress/phase timing and atomically updates same-schema training history after each epoch; it adds no model calls and is not resume support. Engineering equivalence evidence: `C:/Users/UserPc/Desktop/ai/seed211_preparation_20260909_b6e42f/equivalence/comparison.json` (six steps per old/new path, two small epochs, exact losses/order/history/weights/optimizer state). All data used by this test came from the training manifest. This is not a second scientific run. Resource and recovery limitations plus the proposed manual command are in `research/compute_readiness.md`.

Subsequent analysis, only after seed211 is trained and validated: seed113 stays the predetermined reference. Match validation coefficient columns using Hungarian assignment with `1 - Pearson correlation`; if either population standard deviation is <=1e-8, use frozen `2 * MAE` fallback. Apply that permutation unchanged to test outputs, preserving raw numbering, cost/permutation evidence and original artifacts. Use prescribed per-lesion coefficient correlations/MAE/vector L1/dominant agreement, semantic base-2 Jensen-Shannon divergence, nearest top-10 Jaccard and diagnosis consistency, decoded RGB SSIM (range1) and secondary pixel MSE. Average coefficients within the 748 test lesions; paired lesion bootstrap for vector L1/agreement uses 1,000 resamples, seed2026, two-sided percentile95%. No seed-level confidence interval from four comparisons; the full registered matrix is still incomplete. No alignment/test comparison has been run in this task.

### Accepted archetype_stability_v1 matrix

Originally protocol-only; as of 2026-09-13 K5/seed113 and K5/seed211 are completed and validated, with the other eight entries still planned. All entries use accepted immutable split `journal_v1`, 128×128 RGB input, batch 32, 50 epochs, Adam at 0.001, baseline loss weights `1/1/0.01/1`, and no early stopping. Seed 113 is the predetermined reference within each K.

| Experiment ID | K | Latent dimension | Seed | Role | Status |
|---|---:|---:|---:|---|---|
| `stability_v1_K5_seed113` | 5 | 4 | 113 | reference | COMPLETED |
| `stability_v1_K5_seed211` | 5 | 4 | 211 | candidate | COMPLETED |
| `stability_v1_K5_seed307` | 5 | 4 | 307 | candidate | PLANNED |
| `stability_v1_K5_seed401` | 5 | 4 | 401 | candidate | PLANNED |
| `stability_v1_K5_seed503` | 5 | 4 | 503 | candidate | PLANNED |
| `stability_v1_K7_seed113` | 7 | 6 | 113 | reference | PLANNED |
| `stability_v1_K7_seed211` | 7 | 6 | 211 | candidate | PLANNED |
| `stability_v1_K7_seed307` | 7 | 6 | 307 | candidate | PLANNED |
| `stability_v1_K7_seed401` | 7 | 6 | 401 | candidate | PLANNED |
| `stability_v1_K7_seed503` | 7 | 6 | 503 | candidate | PLANNED |

The machine-readable source of this matrix is `research/configs/archetype_stability_v1.json`. Before any run changes to RUNNING, enter the actual implementation commit manually, verify split hashes, and create a unique non-overwriting output directory.

### Gate 1 pipeline smoke

`gate1_pipeline_20260903` completed one epoch over at most 32 rows per partition at K=5, seed 113. It verified manifest loading, one training step, evaluation, coefficient/latent/decoded/nearest exports, checkpoint save and exact reload, completion marking, and unchanged protected-source hashes. It is labeled `scientific_evidence: false`; no metric from it is a stability result. The tracked summary is `research/smoke_tests/gate1_smoke_summary.json`.

### 2026-09-09 validation: stability_v1_K5_seed113

- Status: **COMPLETED; validation PASS WITH LIMITATIONS**. Ran 2026-09-05 through 2026-09-06 on local CPU. Recorded execution revision (not inferred from edits): `0e32c8d469be141d30e627219aeac7ee253a34fa`; recorded pre-run worktree clean.
- Model: unchanged protected May baseline, K=5, latent dimension 4, seed 113; frozen journal_v1 counts 8,011/1,002/1,002; matrix settings above, float32, final epoch, no early stopping, no refinement or relaxation.
- Run: `research/runs/stability_v1/K5/seed_113`; checkpoint: `final.weights.h5`; metrics: `split_metrics.json`; history: `training_history.csv`; coefficients: `validation_coefficients.npy` / `test_coefficients.npy`; raw figures: five `decoded_archetype_raw_A*.png` files in that run directory.
- External log: `C:/Users/UserPc/Desktop/ai/primary_K5_seed113_0e32c8d_20260905_a7f19c.log`.
- Separate validation report: `C:/Users/UserPc/Desktop/ai/validation_K5_seed113_20260909/REPORT.md`; machine-readable checks: sibling `checks.json`; read-only harness: sibling `validate.py`. Original artifacts/log preserved byte-for-byte.
- All 21 inventory hashes/sizes match; 23 total files including inventory/completion. Protected source and all frozen manifest hashes match. Exactly epochs 1–50, all finite. Original checkpoint reload: loaded/passed true, 30 tensors, maximum difference 0. Independent reload succeeds, all tensors finite, fixed-vertex decode matches exactly.
- Validation/test mappings match frozen row order. Coefficients each 1002x5 and latent means each 1002x4 are finite; coefficients nonnegative with maximum row-sum error 1.1920929e-7 (<1e-5). Five decoded archetypes/PNGs, 50 nearest-image identities/ranks and 35 class-table rows checked. No required single-run artifact missing; no alternative checkpoint selected.

Saved final-checkpoint deterministic split metrics:

| Metric | Train | Validation | Test |
|---|---:|---:|---:|
| Total | 296.270875118 | 313.843688488 | 321.530663490 |
| Reconstruction sum | 294.545947618 | 312.109694958 | 319.772646427 |
| Classification CE | 0.898359055 | 0.925765788 | 0.946516316 |
| KL (before 0.01 weight) | 17.987588776 | 17.992367864 | 17.994761825 |
| Simplex | 0.646691086 | 0.628305634 | 0.631548432 |
| Image-weighted pixel MSE | 0.005987259559 | 0.006343114190 | 0.006468955427 |
| Image-weighted accuracy | 0.681063538 | 0.682634731 | 0.678642715 |

Loss definitions: reconstruction sums squared RGB/spatial error per image then averages within each batch; CE is batch-mean sparse softmax cross-entropy; KL sums latent dimensions then averages images; simplex averages squared coordinate distances between fixed vertices and batch-derived z_pred. Runner averages loss scalars equally across batches, including the short final batch. Total = reconstruction + CE + 0.01*KL + simplex, entropy weight 0; arithmetic passes rtol 1e-6 / atol 1e-4. This is the protected baseline's actual optimized objective, not a journal loss change. Pixel MSE instead averages all per-image pixel means; dividing reconstruction_sum by 49,152 does not generally equal it because batches have unequal sizes. Full code-traced audit is in the report.

Epoch-50 training history records stochastic training batches with changing weights: total 292.210280886, reconstruction 290.657267840, CE 0.878302583, KL 17.993002569, simplex 0.494782054. It is not the deterministic final-checkpoint train evaluation above. Epoch-50 validation losses equal final validation losses. No test history, tuning, or performance-based selection.

Recorded interval: UTC `2026-09-05T12:13:44.723683+00:00` to `2026-09-06T02:41:01.421895+00:00`, duration 52,036.698212 s (14h 27m 16.698s). This includes initialization, training, repeated validation, final evaluation/output collection, checkpoint reload and exports; it is not demonstrated uninterrupted training time. Earlier benchmark at `C:/Users/UserPc/Desktop/ai/cpu_feasibility_20260905/RESULTS.md`: 3 warm-ups + 20 timed steps, median 1.517 s (1.309–1.811 s), 251 steps/epoch, 5.29 h extrapolated training-only. Extra work is demonstrated; sustained throttling, contention or sleep remain plausible but unmeasured. No primary phase timing or peak-memory telemetry exists.

`scientific_evidence: null` follows the writer's `False if smoke_test else None`; it neither certifies nor invalidates the run. Determinism warnings are empty, but console notices include oneDNN rounding, INT32 Eigen fallback and unavailable Windows GPU; no traceback/OOM recorded. Validation checks saved artifacts and a fresh reload, not a full raw-image metric reproduction. Execution/integrity pass and no single-run protocol deviation was detected; multi-seed stability is still unanswered. Assess progress/phase logging and interruption recovery before subsequent runs; current final-only checkpointing has no resume. No engineering changes or new training implemented in this task.

## 2026-09-15 — K5/seed401 separate export recovery (D-019)

Retained primary identity: K5/latent4/seed401, journal_v1, unchanged50-epoch training at `e9ef9071453a2d4021470d0586e315463868b5bb`. Original `research/runs/stability_v1/K5/seed_401/` remains FAILED at export, not ordinary completion. Final checkpoint SHA-256 `a97a0e69bb7ec7eb1e0d4bd112dbd3eedc5c9da8f27daf7a9d88b08b52f530fb`. Test raw A2 mass is exactly0; raw A5 is1 throughout held-out rows, all latent means identical, tiny positive masses remain for A1/A3/A4. Distinct decoded vertices do not establish useful usage.

Explicit authorization today permits null semantic probabilities/enrichment/JS with status/reason/mass and defined denominators. Separate recovered exports at `C:/Users/UserPc/Desktop/ai/stability_workflow_local/recovered_exports/K5_seed401_20260915/`; receipt SHA-256 `ca588eb761f759f64f12a369bc64a132a2304cfd3761ca38c8100ae04ef96d8c`. Recovery uses HEAD above plus disclosed uncommitted code hashes and source snapshot. Zero optimizer steps; all1002 validation and1002 test coefficient/latent rows and predictions reproduced exactly, decoded max difference0, per-image MSE within rtol1e-6/atol1e-8. Full validator PASS WITH LIMITATIONS: missing original runtime environment cannot be recreated; recovery timing/environment and copied original determinism warning list are separately attributed. Original checkpoint, configuration, history, outputs, FAILED and logs preserved.

Alignment/test verification at `research/analysis/stability_v1/K5_113_vs_401_workflow_v1/`: validation-only candidate columns `[2,1,4,3,0]`, all25 fallback cells, fixed permutation on test. All five lesion Pearson values are null/false. Semantic JS defined4/5, mean0.4265456054941175 labeled defined pairs only. Lesion-vector L1 mean1.5183502171708312; dominant agreement0.3836898395721925. Existing prescribed lesion bootstrap is retained; no metric-driven tuning/rematching/seed exclusion. Verification summary `research/analysis/stability_v1/recovery_verification_20260915/` deliberately includes only the reference/recovery comparison and is INCOMPLETE, not a full study summary.

Recovery archive: `C:/Users/UserPc/Desktop/ai/stability_workflow_local/archives/recovered_K5_seed401_28c67605830d07eac192c06a.zip`, SHA-256 `ddbad90ceadbc9e4929a83207ff277e028f4b8795aac6be96d4c287f5a563c30`. Same-disk protection only. Engineering report/scripts,50 focused passing tests and native-exit evidence: `C:/Users/UserPc/Desktop/ai/seed401_recovery_20260915/`. Queue was not restarted. Next missing run is K5/503 after user review/version/pin/preflight; this is workflow continuation, not training resume. Multi-seed stability remains unanswered until the full registered comparisons are completed and reviewed.
