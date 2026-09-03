# Gate 1 — Stability infrastructure

Status: **READY FOR PRIMARY TRAINING**.

No primary stability run has been executed. The implemented infrastructure is ready for a future clean, versioned launch; the current CPU-only machine remains an impractical target for ten full runs.

## Accepted design

- Immutable `journal_v1`: 8,011/1,002/1,002 images and 5,975/747/748 lesions for train/validation/test, all seven diagnoses, zero lesion overlap, all 17 split invariants passed.
- Manifest SHA-256: train `4c5a8912d4235f4e91ebbe374113e9db6a9cc919b550b4020b4ce670ecd3acdf`; validation `ac7077667504fdf117d198d87fc01b0922941de37816b3d65947df917fa21deb`; test `7caad8c2d7d92510936d391a8851344d3aa15c68a66ca981c6cd382e2ddb13d7`.
- Primary matrix: K=`{5,7}` x seeds=`{113,211,307,401,503}`; seed 113 reference within each K.
- Validation-only Hungarian matching with `1 - Pearson`; population standard deviation `<= 1e-8` triggers deterministic `2 * MAE` fallback.
- Test-only lesion, semantic, nearest-image, and visual stability evaluation. Visual primary is RGB SSIM at data range 1.0; pixel MSE is secondary.
- Raw four-comparison reporting per K plus mean, sample standard deviation, minimum, and maximum; no four-seed confidence interval. Lesion L1 and dominant agreement also receive the fixed paired bootstrap.

## Infrastructure evidence

- The data adapter builds TensorFlow datasets only from accepted manifests and validates hashes, counts, labels, image existence, stable order, exact coverage, and zero lesion overlap.
- The model adapter verifies and imports the untouched authoritative conference source rather than cloning its model/training logic.
- Fresh-process seed controls cover Python, NumPy, TensorFlow/Keras, deterministic TensorFlow operations, and explicitly seeded training shuffle.
- Primary writes are confined to unique directories below `research/runs/stability_v1/`; smoke writes are confined to `research/smoke_tests/`. Nonempty destinations are refused, and completed reruns receive a new sibling rather than overwriting.
- Synthetic alignment tests cover exact/noisy permutations, degeneracy, mismatched K, and malformed inputs. Model parity, manifest invariants, stable ordering, metrics, and output safety are tested.
- A one-epoch smoke over at most 32 manifest rows per partition completed end-to-end, reloaded the checkpoint exactly, and was labeled non-scientific.
- External GPU preparation and safe artifact collection are documented in `research/external_gpu_execution.md`.

## Exit decision

The scientific and code-level Gate 1 blockers are resolved: the complete suite passed 27/27 tests, the deliberately tiny end-to-end smoke passed, and the exact ten-command matrix dry-run passed without execution. Before a primary launch, the senior developer must review and version these uncommitted changes so that the runner's clean-worktree guard can pass, then provision adequate GPU/storage using the documented procedure. These are operational prerequisites, not permission to run the ten experiments in this gate.

Any future split replacement must be introduced as `journal_v2`; `journal_v1` must remain unchanged. No stability result may compare raw archetype IDs as if they were identifiable.
