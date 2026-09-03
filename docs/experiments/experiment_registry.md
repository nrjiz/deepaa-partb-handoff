# Experiment registry

No experimental runs are registered yet. Copy the template below once per run. Do not overwrite completed entries; update `status` and append dated notes.

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

### Proposed archetype_stability_v1 matrix

Protocol only; no run has started. All entries use proposed split `journal_v1`, 128×128 RGB input, batch 32, 50 epochs, Adam at 0.001, baseline loss weights `1/1/0.01/1`, and no early stopping. Seed 113 is the predetermined reference within each K.

| Experiment ID | K | Latent dimension | Seed | Role | Status |
|---|---:|---:|---:|---|---|
| `stability_v1_K5_seed113` | 5 | 4 | 113 | reference | PLANNED |
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
