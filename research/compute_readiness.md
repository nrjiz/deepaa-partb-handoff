# Gate 1 compute readiness

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
