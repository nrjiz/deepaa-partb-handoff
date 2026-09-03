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

