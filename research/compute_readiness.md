# Gate 1 compute readiness

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
