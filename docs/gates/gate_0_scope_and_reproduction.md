# Gate 0 — Scope and trusted reproduction

Status: **PASS WITH DOCUMENTED LIMITATIONS**.

## Provenance established on 2026-09-03

- Authoritative remote: `https://github.com/Amy007-ali/Deep-Archetypal-Analysis-on-HAM10000.git`.
- Clone: `C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000`.
- Working conference baseline: `main` revision `29acec505161530f9ea811534b05dd52d8df9899` (`2026-05-09T15:15:47+03:00`, `adding results for K=9`). Later preservation commits do not redefine this baseline.
- No tags and no alternative branches exist in the fetched repository.
- HEAD implements TensorFlow 2/Keras probabilistic DeepAA: softmax `a`, fixed regular `Z_fixed`, `mu = a @ Z_fixed`, learned `logvar`, training-time reparameterization, RGB decoder, classification head, KL loss, and batch archetype/simplex loss.
- Four committed May result bundles match the published rounded K=3/5/7/9 metrics, including entropy, sparsity, and coefficient variance.
- The source and README have not changed from commit `661cc7f` through HEAD; commits `6dd8154` and `29acec5` add results only.

Assessment: **Revision `29acec5` is the approved high-confidence working conference baseline, but is not formally proven to be the submitted snapshot.** Exact numerical matching, reconstructed split order, and checkpoint evaluation are sufficient to proceed while retaining the provenance limitation.

## Reproduction completed on 2026-09-03

- Repository-local `docs/` is now canonical; the external docs copy remains preserved.
- SHA-256 inventory covers the source, README, all 226 committed May result files, and all four checkpoints.
- The isolated CPU environment is frozen in `repro/requirements_reproduction.txt`; it is explicitly a current reproduction environment, not a historical lock.
- The baseline image-level split was reconstructed exactly: 8,013 train, 1,001 validation, and 1,001 test images. Manifest hashes are stored in `repro/splits/conference_image_level/split_summary.json`.
- Image IDs are disjoint. Lesion IDs overlap: 357 train/validation, 362 train/test, and 50 validation/test; 22 lesions occur in all three. Across any boundary, 725 unique lesion IDs and 1,737 images are affected.
- Artifact-only verification produced 192 exact numeric matches and 24 correctly rounded usage matches. KL and simplex penalty were correctly identified as unavailable from CSV artifacts alone.
- All four HDF5 checkpoints loaded into the unmodified model. Evaluation at batch size 32 on the reconstructed test order produced 32/32 metric matches within absolute tolerance `1e-5`; the largest observed difference was below `2e-8`.
- The standalone invariant suite passed 30/30 checks: source drift, split sizes and hashes, image-ID disjointness, checkpoint loading, simplex shapes, coefficient nonnegativity/sums, and `mu = a @ Z_fixed`.

## Exit-criteria assessment

- PASS: authoritative repository and a senior-approved working revision identified; formal submission binding remains limited.
- PASS: conference-versus-journal scope inventory reviewed.
- PASS: reproduction environment and exact installed packages documented.
- PASS: existing HAM10000 metadata/image counts and metadata hash validated; no download performed.
- PASS: exact historical split reconstructed and persisted without changing it.
- PASS: model, variational inference, fixed simplex, side information, classifier, and loss definitions verified.
- PASS: all four K checkpoints and all derivable artifact metrics reproduce within tolerance.
- PASS WITH LIMITATION: configuration fields are classified EXPLICIT/INFERRED/UNKNOWN because actual invocation logs do not exist.
- PASS WITH LIMITATION: image IDs are disjoint, but historical lesion-level overlap is quantified rather than absent.
- PASS: 30/30 baseline invariant checks passed without retraining.

## Documented limitations (not Gate 0 blockers)

- No conference tag/release, submitted archive, or author confirmation formally binds the paper to HEAD.
- The conference PDF/source is missing, preventing exact page/table/figure reconciliation.
- Exact historical command lines, package versions, training device, and training histories are absent. Batch size 32, 50 epochs, learning rate 0.001, seed 113, and 0.10/0.10 split fractions remain evidence-backed inferences rather than persisted run records.
- The historical protocol has measured lesion-level overlap. This is preserved as an audit fact; a future lesion-grouped journal split must be separate.
- Training determinism remains unknown because no historical environment/log survives and training `tf.data.shuffle` does not set a local seed.
- The current reproduction environment is modern TensorFlow 2.21/Keras 3.15 on Python 3.13 CPU and is not claimed to match the historical environment.
- Loss finiteness during optimization was not exercised because training was prohibited; deterministic forward/evaluation behavior and checkpoint reload were tested instead.

## Required evidence

The repository path, senior-approved working commit, reproduction environment, dataset metadata hash, split manifests/hashes, reproduction commands, output paths, artifact comparisons, checkpoint comparisons, and invariant results are recorded under `repro/`. Remaining provenance and paper-package evidence should be added if recovered.
