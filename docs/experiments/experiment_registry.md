# Experiment registry

One primary run is registered as completed and validated as of 2026-09-09. Copy the template below once per run. Do not overwrite completed entries; update `status` and append dated notes.

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

### 2026-09-09 preparation: stability_v1_K5_seed211

Status remains **PLANNED / NOT STARTED**. Intended output: `research/runs/stability_v1/K5/seed_211`; intended external log: `C:/Users/UserPc/Desktop/ai/primary_K5_seed211_20260909_b6e42f.log` (both absent at preparation). Execution revision is **pending user review/commit**, not the preparation HEAD `42ec30f84cf826a2a3465fbb0778af4b04fc0bd5` or seed113's execution revision. No metric-driven tuning: K5/latent4/seed211, journal_v1, RGB128 float32, batch32, 50 epochs, Adam0.001, reconstruction1 + CE1 + KL0.01 + simplex1; no early stopping, entropy training term or temperature scaling; final epoch checkpoint and existing deterministic controls.

Research-only instrumentation records flushed progress/phase timing and atomically updates same-schema training history after each epoch; it adds no model calls and is not resume support. Engineering equivalence evidence: `C:/Users/UserPc/Desktop/ai/seed211_preparation_20260909_b6e42f/equivalence/comparison.json` (six steps per old/new path, two small epochs, exact losses/order/history/weights/optimizer state). All data used by this test came from the training manifest. This is not a second scientific run. Resource and recovery limitations plus the proposed manual command are in `research/compute_readiness.md`.

Subsequent analysis, only after seed211 is trained and validated: seed113 stays the predetermined reference. Match validation coefficient columns using Hungarian assignment with `1 - Pearson correlation`; if either population standard deviation is <=1e-8, use frozen `2 * MAE` fallback. Apply that permutation unchanged to test outputs, preserving raw numbering, cost/permutation evidence and original artifacts. Use prescribed per-lesion coefficient correlations/MAE/vector L1/dominant agreement, semantic base-2 Jensen-Shannon divergence, nearest top-10 Jaccard and diagnosis consistency, decoded RGB SSIM (range1) and secondary pixel MSE. Average coefficients within the 748 test lesions; paired lesion bootstrap for vector L1/agreement uses 1,000 resamples, seed2026, two-sided percentile95%. No seed-level confidence interval from four comparisons; the full registered matrix is still incomplete. No alignment/test comparison has been run in this task.

### Accepted archetype_stability_v1 matrix

Originally protocol-only; as of 2026-09-09 the K5/seed113 reference is completed and validated, with the other nine entries still planned. All entries use accepted immutable split `journal_v1`, 128×128 RGB input, batch 32, 50 epochs, Adam at 0.001, baseline loss weights `1/1/0.01/1`, and no early stopping. Seed 113 is the predetermined reference within each K.

| Experiment ID | K | Latent dimension | Seed | Role | Status |
|---|---:|---:|---:|---|---|
| `stability_v1_K5_seed113` | 5 | 4 | 113 | reference | COMPLETED |
| `stability_v1_K5_seed211` | 5 | 4 | 211 | candidate | PLANNED |
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
