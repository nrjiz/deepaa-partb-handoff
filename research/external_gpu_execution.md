# External GPU execution for stability_v1

This is an execution procedure, not a separate scientific implementation. The same versioned `research/tools/run_stability_one.py` runner, frozen manifests, configuration, alignment code, and metrics must be used locally and externally.

## Execution hold and provider-independent plan (2026-09-05)

No usable GPU target has been established. No provider is selected or authorized for provisioning. The commands below are preparation instructions only. The first eventual run is K=5, latent dimension 4, seed 113; do not launch the matrix.

**Source-integrity blocker:** the protected Windows working source has CRLF bytes with SHA-256 `6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d`. The committed LF blob has SHA-256 `a3896fcb4648cf9b709fcfcdac081f0d47a1c16c7685e84c1551e5b82ab16a8c`. They are otherwise byte-identical after in-memory newline normalization, but `model_adapter.py` and `runner.py` check literal bytes. A default Linux checkout is rejected. Resolve checkout-byte preservation or a narrowly reviewed integrity-code amendment explicitly before execution; no such change is implemented here. Never substitute a hash ad hoc. Any code amendment must be versioned by the senior developer and truthfully recorded under its new revision.

### Local bundle and isolated restoration

An existing complete-history bundle already preserves research commit `7911e8643495a2f34820ca0adedbea1f322ab8da` and conference baseline `29acec505161530f9ea811534b05dd52d8df9899`:

- Path: `C:\Users\UserPc\Desktop\ai\gpu_handoff_7911e86\deepaa_journal_research_7911e86.bundle`
- SHA-256: `d3a8ebfcc8441c414c0abf3fc36994251a6e64876bc51ff545bdd7ac65b3592d`
- Size: 110,993,619 bytes; verification passed again on 2026-09-05.
- Bundle-only verification clone already discovered all 27 tests. Its LF source mismatch is recorded in that folder's `LOCAL_VERIFICATION.txt`.

Reuse this exact bundle when appropriate; no new large bundle was created in this audit. For a future recreation of the same revision, first confirm both refs still have the expected IDs and use a NEW, nonexistent destination, for example the following PowerShell commands from the canonical repository (not executed here):

```powershell
if (Test-Path -LiteralPath 'C:\Users\UserPc\Desktop\ai\gpu_handoff_7911e86\deepaa_7911e86_recreated.bundle') { throw 'Destination already exists' }
git bundle create C:/Users/UserPc/Desktop/ai/gpu_handoff_7911e86/deepaa_7911e86_recreated.bundle journal-research-local main
git bundle verify C:/Users/UserPc/Desktop/ai/gpu_handoff_7911e86/deepaa_7911e86_recreated.bundle
Get-FileHash -Algorithm SHA256 C:\Users\UserPc\Desktop\ai\gpu_handoff_7911e86\deepaa_7911e86_recreated.bundle
```

Run sequentially and stop on any failure. Bundles contain committed objects, not uncommitted documentation, ignored runs, the Python environment, or HAM10000 JPEGs. The public repository must not be assumed to contain the research commit.

After a separately authorized transfer, restore into an unused isolated directory on the chosen machine. Example Bash paths are placeholders, not a selected host:

```bash
sha256sum /transfer/deepaa_journal_research_7911e86.bundle
git -c core.autocrlf=false clone --branch journal-research-local /transfer/deepaa_journal_research_7911e86.bundle /work/deepaa_7911e86
cd /work/deepaa_7911e86
git config remote.origin.pushurl DISABLED
git remote -v
git rev-parse HEAD
git cat-file -t 29acec505161530f9ea811534b05dd52d8df9899
git status --short
sha256sum daa_ham10000_interpretability_fixed_v2.py research/splits/journal_v1/train.csv research/splits/journal_v1/validation.csv research/splits/journal_v1/test.csv
```

Require the recorded bundle hash, expected HEAD, baseline commit presence, clean status, disabled push, and exact source/manifest hashes. This LF restoration intentionally exposes the known source mismatch: STOP there until the approved portability resolution is applied and validated. Git remote configuration is not included in a bundle; disabling pushes in the new clone is essential. No canonical refs or remote configuration need changing.

### Data and environment inputs

Transfer the two JPEG directories separately only when authorized, retaining basenames `HAM10000_images_part_1` and `HAM10000_images_part_2`, original case-sensitive `<image_id>.jpg` names, and all 10,015 files. The runner uses the three tracked manifests as metadata; `HAM10000_metadata.csv` is not separately required by this entry point. Parent image paths are supplied with `--image-dirs`; no dataset environment variable is used. Manifest flags default to the checkout's `research/splits/journal_v1/`. No Windows-only runtime data path was found.

Before a future data transfer, create an external per-JPEG SHA-256 inventory (relative source directory, filename, size, hash) and verify it at the destination, along with the archive checksum if archived. The current adapter checks manifest identity and JPEG existence, not JPEG content hashes. No such inventory or dataset copy was created in this audit. Metadata/manifests alone cannot establish that transferred image bytes match.

Use the existing pinned requirements and validated version record below. A wider compatible Python range and target CUDA/cuDNN installation are not established by local validation. Keras is transitive rather than directly pinned; record all resolved versions and run `python -m pip check`. Keep environments, credentials, transferred archives, and console logs outside the clean checkout. Existing compute guidance uses 8 GiB GPU memory as an unbenchmarked operational target. Budget approximately 2.70 GiB for JPEGs, repository/history and environment space, and at least 1 GiB run/archive headroom (expected fixed K5 outputs are tens of MB; full-run size and peak GPU RAM have not been measured).

The three data-dependent tests expect `~/Downloads/archive/HAM10000_images_part_*`; otherwise they skip. Supply that layout or approved external links and require all 27 tests to pass on the target. Independently validate configurable data paths using the manifest-only command below. Do not edit test code or silently accept skipped data checks as validation.

### First-run readiness and validation

Before primary training:

1. Resolve the source-byte issue and any target dependency incompatibility with explicit review; require a clean approved revision, truthful HEAD, disabled pushing, exact source and manifest hashes, and no existing target run directory.
2. Require a TensorFlow-visible GPU, record physical model/driver, CUDA/cuDNN build information, RAM/VRAM and free disk, and verify deterministic-op support. The runner itself does not require a GPU, so this external check is mandatory. Choose enough uninterrupted runtime and durable storage for 50 epochs; runtime is not yet benchmarked.
3. Verify all data hashes/membership, 8,011/1,002/1,002 rows, zero lesion overlap, and frozen mapping order. Revalidate the existing tests in the eventual environment; do not repeat local smoke training for the known CPU limitation.
4. Confirm the existing constants: K=5, latent_dim=4, seed=113, 128x128 RGB, batch 32, 50 epochs, Adam 0.001, no early stopping, final checkpoint, weights reconstruction/classification/KL/simplex `1/1/0.01/1`, no entropy objective or temperature scaling.

Existing CLI discovery and read-only preflight commands (there is no single-run `--dry-run`, `--preflight`, or resume flag):

```bash
python research/tools/run_stability_one.py --help
python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print(g,tf.sysconfig.get_build_info()); assert g, 'STOP: no TensorFlow GPU'; tf.config.experimental.enable_op_determinism()"
python -B -c "from pathlib import Path; from research.code.data import load_journal_v1; p=Path('research/splits/journal_v1'); f=load_journal_v1(p/'train.csv',p/'validation.csv',p/'test.csv',[Path('/data/HAM10000_images_part_1'),Path('/data/HAM10000_images_part_2')]); print({k:len(v) for k,v in f.items()})"
python -B -m pytest research/tests --collect-only -q -p no:cacheprovider
```

The single-run command under 'Launch one run' below is the exact supported invocation. Its manifest defaults are authoritative; explicit supported overrides are `--train-manifest research/splits/journal_v1/train.csv --validation-manifest research/splits/journal_v1/validation.csv --test-manifest research/splits/journal_v1/test.csv`. Do not execute it until the hold is cleared. Do not use `--allow-completed-rerun` or launch other seeds.

After execution, require `COMPLETED.json`, no failure/running marker, history epochs exactly 1 through 50, matching configuration/HEAD/hashes, and successful `checkpoint_reload.json` (the runner reloads and compares all weights). Independently load `final.weights.h5` in the approved model for transfer validation. Require finite numeric losses/metrics, coefficients and latent means; validation/test coefficient shape `(1002,5)`, latent shape `(1002,4)`, nonnegative coefficients and row sums approximately one (e.g. absolute tolerance `1e-5`), and mappings identical to frozen image_id/lesion_id/dx order. Verify all artifact inventory sizes and SHA-256 values. Require decoded raw array plus five PNGs, top-10 identities per archetype, class/enrichment table, split metrics for train/validation/test, environment, warnings and completion metadata. Inventory is written before completion, so `COMPLETED.json` and the inventory itself need transport checksums separately. Recheck protected hashes and historical results. No metric quality threshold determines retention or success; no alignment or stability conclusion follows from one run.

### Interruption and return limitations

`configuration.json` is saved before training. History accumulates in memory and is written only after all epochs, followed by the final checkpoint. There are no intermediate checkpoints, optimizer-state resume files, or resume CLI. On an ordinary caught exception, `FAILED.json` is written and `.RUNNING` removed. Abrupt termination or KeyboardInterrupt can leave `.RUNNING` and whatever files already reached disk; ephemeral host loss can lose all local files. Mid-training progress is not recoverable with this runner. Do not rerun into the same partial directory, delete it, or claim resume support. Preserve failures and seek a separately authorized new attempt.

Save stdout/stderr to a new external log path: `warnings.json` captures determinism warnings, not all native TensorFlow messages. The recorded duration covers training plus export/reload, not training alone. Checkpoint/metadata persistence changes, if needed for the chosen target, require a reviewed new revision before launching.

After validation, package the complete run directory and external console/environment/GPU records into a new archive outside the checkout; preserve failed attempts similarly with their failure status. Hash the archive before transfer and after return. Extract first into a new isolated verification directory, verify all inventory hashes and mappings/checkpoint, and copy only to a nonexistent destination under `research/runs/stability_v1/`. Never merge attempts or overwrite historical outputs. No package, transfer, upload, installation, or primary execution is performed by this plan.

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
