# Gate 1 compute readiness

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
