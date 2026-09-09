# Gate 1 compute readiness

## Seed211 instrumentation and preparation: 2026-09-09

**Preparation only; no primary launch.** Initial HEAD `42ec30f84cf826a2a3465fbb0778af4b04fc0bd5` was clean, with no existing user changes. The previous validation documentation is committed. The new execution revision is not known until the senior developer reviews/commits these unstaged changes. All clean-worktree/source/manifest/configuration/output guards remain intact. Protected source, frozen manifests, completed seed113 artifacts and original external log are unchanged.

### Instrumentation

`research/code/runner.py` now uses `research/code/progress.py` for flushed JSON console messages plus a new `progress.jsonl` in the exclusively reserved active run directory. Preflight/loading records are buffered until reservation; failures before reservation still appear on the console. Phase start/end records cover manifest loading, initialization/dataset setup, training, per-epoch validation/history, checkpoint export/reload, final split-loss evaluation, output collection and artifact export. Durations/elapsed values use perf_counter; UTC timestamps are included. These are elapsed intervals, not active CPU time, and can include scheduling/idle effects.

Each epoch logs start/end, number out of 50, duration, already-computed train/validation losses and elapsed time. Batch progress occurs at batch1, each multiple of50 and the final batch (251 for primary training). No extra inference or RNG access is added. Remaining-loop ETA is mean completed-epoch duration times remaining epochs; it includes repeated validation/history and explicitly excludes final evaluation/export overhead. It is an approximation, not a finish-time promise.

After every completed epoch, a temporary CSV is atomically replaced into `training_history.csv` using the original final columns/order/float formatting. Only the owned active run history is replaced; completed/nonempty runs still refuse reuse. This preserves completed-epoch observations but not model/optimizer state. File close/atomic replacement is not a power-loss durability guarantee; an interrupted write may leave a temporary file. `progress.jsonl` is closed before the existing inventory writer hashes it. No progress-file writes occur after sealing; the final completion message is console-only. Other final artifact schemas and the model/evaluation call order remain unchanged.

### Verification and resource evidence

Evidence root: `C:/Users/UserPc/Desktop/ai/seed211_preparation_20260909_b6e42f/`. `equivalence/comparison.json` reports exact equality between loop statements extracted from committed pre-change runner and instrumented runner. Fresh sequential seed211 workers each performed two miniature epochs over 65 training images (32/32/1), with 10 other training-manifest images used for engineering evaluation. All six step-loss vectors, 30 final model tensors, 62 optimizer variables (including iteration6), batch-data hashes and final histories match with maximum difference0. This is stricter than existing numerical tolerances; epoch-1 history was verified before epoch 2 and progress/ETA records checked. No held-out image performance was inspected. Worker logs, states, summaries and incremental histories are retained separately from primary outputs.

The test directly exercises extracted training-loop statements; it does not invoke a primary runner or weaken its guards. It is not a full epoch, benchmark, long-run proof, cross-device guarantee or repeat of final image exports. The extracted block includes the following checkpoint-phase announcement, but no checkpoint save/reload is executed by this loop test. Unit tests separately cover phase output, incremental history, refusal of an existing progress file and inclusion in the checksum inventory. The full research suite passed **29/29 tests** in 53.00 s, with 56 dependency deprecation warnings, no failures. Results are recorded in `pytest.xml` and `REPORT.md` in the evidence root. `git diff --check` passed; all seven changed/new repository files remain unstaged.

`preflight.json` verifies the protected source and three manifests, all 8,011/1,002/1,002 image paths, the unchanged seed113 run/log against prior validation hashes, interpreter and unused intended output/log. Discovered dataset folders are directly under archive: `HAM10000_images_part_1` and `HAM10000_images_part_2`. This task did not repeat the all-image JPEG decoding benchmark; the bounded test decoded its 75 training images through existing preprocessing. Interpreter: `C:/Users/UserPc/Desktop/ai/.venvs/deepaa_data2026_repro/Scripts/python.exe`, Python3.13.3, TensorFlow2.21.0, Keras3.15.1, CPU only; no dependency/environment changes.

Resource snapshots: 15.85 GiB physical RAM; only about0.88 GiB available before the bounded test, about0.59 GiB at one in-test inspection, and1.96 GiB in the saved preflight snapshot (87% memory load). Disk free in that snapshot:25,354,584,064 bytes (23.61 GiB). No unrelated Python training process was observed before testing; worker execution was sequential. Memory availability is an operational limitation, not a protocol issue. Recheck RAM/disk, stable power and absence of competing training immediately before launch; do not terminate user processes or lower frozen settings. The seed11314h27m interval is planning evidence, not a seed211 runtime guarantee.

### Interruption recovery assessment — not implemented

Exact epoch-boundary continuation would need all model variables; Adam slot variables, iteration and learning-rate state; completed epoch and history; Python/NumPy RNG state; the actual TensorFlow/Keras random-op state/counters used by initialization and protected `tf.random.normal` posterior sampling; and dataset shuffle iteration/order state. The current seeded tf.data shuffle uses reshuffle_each_iteration=True with buffer min(N,5000), parallel decode/prefetch and partial batches retained. Restarting the seed or skipping an arbitrary number of batches is not evidence that the next epoch order/random samples match. The protected model has no dropout or BatchNorm moving-statistic layers; z_pred is batch-derived, not a separate persistent parameter, but all model variables must still be captured.

Proposed separately reviewed approach: define a transaction boundary after epoch validation/history, save a versioned recovery bundle with model+optimizer checkpoint, epoch/global-step/history, manifests/source/configuration/environment provenance and verified RNG/iterator state; publish it atomically in a non-overwriting recovery location. Do not assume tf.train.Checkpoint automatically captures this runner's legacy stateful tf.random.normal counters or the reshuffling sequence across newly constructed epoch iterators. Establish a supported capture/restore strategy first; if exact preservation requires changing random/data semantics, that needs separate review, not a silent implementation here. Test interrupted-versus-uninterrupted continuation across epoch boundaries for exact next-batch order, random draws, losses, optimizer state and weights, plus incomplete-write/refusal cases. Weights-only saves are not equivalent continuation. No resume implementation is authorized or included.

### Proposed subsequent Windows CMD command — DO NOT RUN YET

The user must review and commit these changes, then perform final clean-revision preflight and manually launch only after authorization. Verify the new HEAD and clean `git status --short --untracked-files=all`; do not substitute the preparation HEAD as the execution revision. Both proposed paths below were absent during preparation; recheck at launch. Keep the reference run untouched. The CMD existence checks prevent ordinary reuse, not a concurrent-launch race; ensure one launcher only.

```bat
cd /d "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000"
set "PYTHONHASHSEED=211"
set "TF_DETERMINISTIC_OPS=1"
set "PYTHONDONTWRITEBYTECODE=1"
if exist "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\runs\stability_v1\K5\seed_211" (
  echo STOP: intended output already exists.
) else if exist "C:\Users\UserPc\Desktop\ai\primary_K5_seed211_20260909_b6e42f.log" (
  echo STOP: intended log already exists.
) else (
  "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u research\tools\run_stability_one.py --k 5 --seed 211 --output-dir "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\runs\stability_v1\K5\seed_211" --image-dirs "C:\Users\UserPc\Downloads\archive\HAM10000_images_part_1" "C:\Users\UserPc\Downloads\archive\HAM10000_images_part_2" > "C:\Users\UserPc\Desktop\ai\primary_K5_seed211_20260909_b6e42f.log" 2>&1
)
```

No early stopping/entropy objective/temperature scaling or checkpoint selection change: K5/latent4/seed211, journal_v1, RGB128 float32, batch32, 50 epochs, Adam0.001, reconstruction1 + CE1 + KL0.01 + simplex1, final epoch and existing deterministic controls. Incremental history is not permission to resume a failed attempt. After completion and separate validation, use seed113 reference and validation-only Hungarian matching with frozen Pearson/degeneracy costs, then the unchanged test mapping and prescribed lesion aggregation/bootstrap/semantic/nearest/visual metrics. Preserve raw outputs and do not perform comparison now; see the registry for the analysis plan.

## First primary CPU outcome: 2026-09-09

The older blocked/readiness statements below are historical snapshots. Explicitly authorized local CPU execution has now completed one primary run: `research/runs/stability_v1/K5/seed_113`, revision `0e32c8d469be141d30e627219aeac7ee253a34fa`, 50 epochs with unchanged K5/latent4/seed113, RGB128, batch32, Adam0.001, float32 and baseline objective. Validation verdict **PASS WITH LIMITATIONS**: inventory/source/manifests match, history complete/finite, checkpoint reload and required raw exports pass. See `../docs/experiments/experiment_registry.md` for metrics/definitions and `C:/Users/UserPc/Desktop/ai/validation_K5_seed113_20260909/REPORT.md` plus sibling `checks.json` for the full audit. External log: `C:/Users/UserPc/Desktop/ai/primary_K5_seed113_0e32c8d_20260905_a7f19c.log`. All original evidence is unchanged.

### Benchmark versus recorded run interval

Engineering benchmark `C:/Users/UserPc/Desktop/ai/cpu_feasibility_20260905/RESULTS.md` completed 3 warm-ups and 20 timed steps: median 1.517 s, range 1.309–1.811 s, finite losses. All 8,011 training images resolved/read successfully. Initialization/readability audit took 329.675 s, first batch wait 0.150 s and supervisor total 370.656 s. Worker peak working set was 908.10 MiB; minimum sampled available RAM 2.94 GiB, memory load 81%; no allocation failure. These are benchmark-only memory observations. Its 251 steps/epoch (last batch 11) yielded 6.35 training-only minutes/epoch and 5.29 h for 50 epochs; the 4.56–6.31 h min/max sensitivity range was not a confidence interval. Filesystem caches had been warmed by the readability audit.

Primary runtime_environment.json records UTC `2026-09-05T12:13:44.723683+00:00` through `2026-09-06T02:41:01.421895+00:00`, exactly **52,036.698212 s = 14h 27m 16.698s** (about 2.73 times the training-only benchmark estimate). Runner datetime wall-clock timing starts after manifest loading/output reservation, before initialization; it ends after training, per-epoch validation, final checkpoint reload, split-loss evaluation, separate output collection and exports. It excludes earlier preflight and final marker/inventory writes. No file-timestamp estimate is substituted for this record.

The benchmark omitted demonstrated evaluation/export/reload work and sampled only 20 warmed steps. There are no primary epoch/phase timings or active-process CPU-time measurements to apportion the gap. Suspension, contention, thermal/power throttling or memory pressure are plausible but not established. Do not equate the full recorded interval with uninterrupted training, or apply this K5 duration to K7 as a measured estimate. Console fallback/CPU notices did not prevent completion; empty warnings.json covers determinism warnings only. Primary peak memory was not recorded.

### Requirements before subsequent execution

One CPU run is demonstrably possible, but a ten-run local schedule is not established as practical. Senior developer/BRAIN should assess sustained compute, stable power, fresh RAM/disk headroom and output retention, then separately authorize any next fixed-seed run. Existing runner already works on CPU and has no explicit GPU-required code guard; preserve clean-worktree, configuration, source/manifest and non-overwrite guards. No primary runner/device change was made here.

Assess progress/phase logging, durable epoch records and interruption recovery as separate engineering work before another long run. Current history is held in memory until all epochs finish; only final weights are saved, without resume. An interruption can lose hours of progress. A future recovery design would need reviewed optimizer/RNG/data-order handling, not just weight loading. No logging, checkpoint or resume changes were implemented in this validation task. Completed outputs must never be overwritten. The senior developer handles Git; these documentation edits remain unstaged and must be reviewed before a guarded launch.

Scientific stability remains unanswered until independent runs are aligned on validation and compared under the frozen test protocol. The current metrics and scientific_evidence=null do not themselves settle that question.

## Follow-up audit: 2026-09-05

Primary execution is currently blocked. Windows hardware queries report Intel UHD Graphics 620 (driver `31.0.101.2125`, dated 2023-05-24) and three `MS Idd Device` indirect-display adapters. No NVIDIA adapter was listed, and `nvidia-smi` was not found on PATH. This is a hardware inventory finding independent of TensorFlow's empty GPU list; Intel graphics are physically present. WSL commands report that WSL is not installed.

The machine runs Windows 11 Pro, 64-bit, build `10.0.26200`. OS-visible RAM was 16,616,576 KiB (15.85 GiB), with 4,726,712 KiB (4.51 GiB) free at inspection. C: had 12,783,194,112 bytes (11.91 GiB) free. These free-memory/storage figures are snapshots.

The inspected interpreter was `C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe`: Python 3.13.3, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.2. TensorFlow reports no GPU and a non-CUDA/non-ROCm build. Its determinism API exists; support for the eventual GPU workload still requires target validation. The default PATH Python is a different installation at `C:\Users\UserPc\AppData\Local\Programs\Python\Python313\python.exe`. No GPU target/access details are established in project documentation. No provider was selected.

The initial repository was clean at `journal-research-local`, HEAD `7911e8643495a2f34820ca0adedbea1f322ab8da`; local `main` and locally cached `origin/main` both matched `29acec505161530f9ea811534b05dd52d8df9899`. Fetch URL remained Ali's documented repository and push URL remained `DISABLED`. No live remote verification was performed. Protected-source and all three manifest hashes matched. `research/runs/` contained only its `.gitignore`, with no primary/partial/completed runs.

The existing bundle under `C:\Users\UserPc\Desktop\ai\gpu_handoff_7911e86` was reverified without creating another bundle. Repository files including `.git` and ignored artifacts totaled 227,375,098 bytes; Git pack storage was 105.46 MiB; the bundle is 110,993,619 bytes. There is sufficient local headroom for that existing handoff, but no local usable GPU target for the required run. Dataset storage was previously measured at approximately 2.70 GiB. See `external_gpu_execution.md` for the transfer plan, known LF/CRLF integrity blocker, interruption limitations, and first-run checklist.

These audit documentation edits leave the canonical worktree dirty. The senior developer must handle them before any guarded primary launch; do not bypass the guard. No installation, transfer, training, or smoke test was performed during this audit.

Inspected 2026-09-03 without primary training or benchmarking. A 32-row-per-partition, one-epoch smoke run was completed solely to validate the pipeline and is not scientific evidence.

- TensorFlow environment: `C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro`
- Python/TensorFlow/SciPy: Python 3.13.3, TensorFlow 2.21.0, Keras 3.15.1, SciPy 1.18.1
- TensorFlow-visible GPU devices: none
- TensorFlow-visible CPU device: `/physical_device:CPU:0`
- Processor: Intel64 Family 6 Model 142 Stepping 10
- Logical CPU count: 8
- Disk containing the repository: 510,860,980,224 bytes total; 13,695,135,744 bytes free at inspection (about 12.75 GiB)

The environment is suitable for model import, checkpoint evaluation, split preparation, and small tests. It is not a practical default for the ten planned 50-epoch 128×128 training runs: native-Windows TensorFlow is CPU-only here, and free disk headroom is low for isolated multi-run checkpoints, coefficients, logs, and figures.

Before local training, free substantial disk space and estimate output retention. The documented external-GPU route is `research/external_gpu_execution.md`; it requires a clean versioned checkout and verifies source/manifests before execution. Use a compatible TensorFlow environment with deterministic-operation support where possible, a CUDA-capable GPU (8 GiB or more GPU memory is a reasonable operational target for this model/batch size), adequate system memory, and enough persistent storage for the dataset plus ten non-overwriting run directories. Move neither data nor credentials until separately authorized. No GPU stack was installed in this task.
