# External GPU execution for stability_v1

This is an execution procedure, not a separate scientific implementation. The same versioned `research/tools/run_stability_one.py` runner, frozen manifests, configuration, alignment code, and metrics must be used locally and externally.

## Environment

Use a clean Linux GPU host or Colab runtime with a CUDA-capable GPU and a TensorFlow build compatible with that runtime. The infrastructure was validated with Python 3.13.3, TensorFlow 2.21.0, NumPy 2.5.2, pandas 3.0.5, h5py 3.14.0, SciPy 1.18.1, and pytest 9.1.1. If the external platform requires a different supported Python/TensorFlow pair, record the exact versions and rerun all infrastructure tests plus the smoke test before primary training.

```bash
python -m venv /path/outside/repository/deepaa-stability-v1
source /path/outside/repository/deepaa-stability-v1/bin/activate
python -m pip install -r research/requirements_stability_v1.txt
python -m pytest research/tests -q
```

The primary runner refuses a dirty worktree. Use a reviewed commit containing the infrastructure and record that commit in every run configuration. Do not modify the protected conference source.

## Dataset and frozen manifests

Make the already-authorized HAM10000 files available as two directories named `HAM10000_images_part_1` and `HAM10000_images_part_2`. Do not regenerate splits externally. Copy the repository's tracked `research/splits/journal_v1/*.csv` files with the code; the runner verifies these SHA-256 values before every run:

- train: `4c5a8912d4235f4e91ebbe374113e9db6a9cc919b550b4020b4ce670ecd3acdf`
- validation: `ac7077667504fdf117d198d87fc01b0922941de37816b3d65947df917fa21deb`
- test: `7caad8c2d7d92510936d391a8851344d3aa15c68a66ca981c6cd382e2ddb13d7`

Run the non-scientific smoke path first and verify its `COMPLETED.json`. Smoke output belongs under `research/smoke_tests/`, never under primary runs.

## Launch one run

From the repository root:

```bash
python research/tools/run_stability_one.py \
  --k 5 \
  --seed 113 \
  --output-dir research/runs/stability_v1/K5/seed_113 \
  --image-dirs /data/HAM10000_images_part_1 /data/HAM10000_images_part_2
```

K and seed are restricted to the pre-registered matrix. Scientific training settings are not exposed as CLI overrides. The wrapper sets process-level determinism variables before TensorFlow import; the runner also records Python, NumPy, TensorFlow/Keras, dataset-shuffle, initialization, and posterior-sampling seeds.

## Launch the predetermined matrix

Inspect commands without executing:

```bash
python research/tools/run_stability_matrix.py \
  --image-dirs /data/HAM10000_images_part_1 /data/HAM10000_images_part_2
```

Only after senior review/versioning and GPU provisioning, launch exactly those ten commands with explicit `--execute`. The matrix is K=`{5,7}` × seeds=`{113,211,307,401,503}`. Seed 113 is the reference within each K.

## Alignment and held-out evaluation

After runs complete, learn each permutation from validation outputs:

```bash
python research/tools/align_validation_coefficients.py \
  --reference-run research/runs/stability_v1/K5/seed_113 \
  --candidate-run research/runs/stability_v1/K5/seed_211 \
  --output-dir research/analysis/stability_v1/K5/seed_113_vs_211/alignment
```

Then calculate test-only stability metrics:

```bash
python research/tools/evaluate_stability_pair.py \
  --reference-run research/runs/stability_v1/K5/seed_113 \
  --candidate-run research/runs/stability_v1/K5/seed_211 \
  --alignment-dir research/analysis/stability_v1/K5/seed_113_vs_211/alignment \
  --output-dir research/analysis/stability_v1/K5/seed_113_vs_211/test_metrics
```

Do not use test artifacts for alignment.

## Collect outputs safely

Copy each complete run directory back to its exact empty destination under `research/runs/stability_v1/K*/seed_*`. Refuse a transfer when that destination already exists. After transfer, require `COMPLETED.json`, compare the recorded manifest/source hashes, verify the recorded Git commit and environment, and reload the final checkpoint before analysis. Never merge two run directories and never copy primary outputs into `Results_HAM_DAA` or `repro/`.

No data upload, external connection, or external computation was performed while preparing these instructions.
