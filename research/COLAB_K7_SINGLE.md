# Independent K7/latent6/seed113 preparation (D-021)

ST-04A review update (2026-10-13): the K7 post-training worker now routes only
K7/latent6/seed113 through the schema-scoped adapter described in
`COLAB_K7_VALIDATION.md`. Shared D-019-bound validation and scientific code remain
unchanged. New complete-worker synthetic regressions are included in the target
orchestration cell. This uncommitted checkout may only produce a disabled draft;
a new reviewed clean commit/release and fresh Colab readiness are required before
any primary launch. No readiness or K7 scientific result is established locally.

This supersedes the six-run Colab notebook instructions for this release. It
prepares ONE frozen K7 reference run, seed113, from epoch 1. The frozen config
and unchanged model_adapter define latent_dimension = K - 1: K7 means latent6,
not latent4. The notebook requires explicit senior acknowledgement of latent6
before launch. K5/seed503 belongs exclusively to Windows. No Windows completion
is required and no live Windows registration, process, lock or output is read.

All development is in `colab_k7_isolated_20260922`, cloned from approved Windows
commit `2eecc13643bf56cdcd9d0c11f91293f46d87d024`. Never merge/check out these
edits into the active Windows source during training. No scientific code,
split bytes, D-019-bound sources, tolerances or training settings changed.

## Restoration repair

Actual base release: `45c6b81b9fdc240476f5b6d21d4073f9f6b760c4`, ZIP SHA-256
`23e5e25186d83376a480ce3a1d7a343b418cf234c6a43428d00cb98931c3b856`.
Its packaged notebook has 244 CRLF endings plus one LF ending. A uniform LF or
CRLF checkout cannot reproduce that file's size before the exact byte overlay.
Git's stat shortcut can then report a modification despite identical filtered
blob content. Ordinary `update-index --refresh` alone does not resolve it.

Restoration derives per-file checkout EOL rules from the verified snapshot.
After exact overlay, any remaining diff-files entries must be packaged source
files whose SHA-256 matches the snapshot and whose Git filtered blob ID matches
the reviewed commit. Only their disposable index stat-size fields are invalidated
to zero (index v2 with verified SHA-1 checksum and exclusive index.lock). Git then
performs `update-index --refresh`. All index paths/modes/blob IDs/stages are
compared with the commit before and afterward; working-tree bytes are rehashed.
No content is staged, no source reset/normalization occurs, and no ignored path,
assume-unchanged bit or weakened cleanliness/hash gate is used. Unsupported index
formats, checksum changes, real edits or an existing index lock stop restoration.
Final Git cleanliness, exact source/manifests, historical aliases and unchanged
D-019 registration/receipt/archive checks remain mandatory.

## Build without touching active Windows execution

Use `research/tools/repackage_colab_k7.py`, NOT the old builder that reads live
Windows workflow state. Inputs are this isolated checkout and the immutable
`colab_migration_release_restore_fix_20260922T132355616693Z` directory. It verifies
the original ZIP/receipt, copies its evidence/preservation/data/relocation bytes,
checks D-019 code bindings, and packages the new reviewed source and Git bundle.
The new manifest records the base release hash, evidence date, current source
HEAD and exact one-run scope. Those historical K5 records are not claims about
the current Windows attempt. Omit --approved-head only for a disabled review
draft. Release builds require the explicitly reviewed clean commit. New output
directories only; no overwrites, Windows pin/lock edits or Git publication.

## Notebook procedure

1. Upload the rebuilt notebook, ZIP and matching TRANSFER_RECEIPT.json together.
   Compare the receipt SHA-256 to the senior build output.
2. Use the one project Colab session under the accepted manual discipline.
   Select GPU. Kernel bytecode writes are disabled before research imports.
   A cached `research` module stops setup: restart the kernel to clear imports,
   preserving files and attempts. Do not manually reload mixed-release modules.
3. The initial cell checks PAYLOAD/WORKSPACE absence before extraction. Defaults
   are `/content/deepaa_transfer_k7` and `/content/deepaa_workspace_k7`. If these
   contain a failed setup, preserve them and explicitly choose new absent paths.
   Never delete/reuse a partial checkout or edit PACKAGE.json to force a pass.
   Keep DURABLE fixed at `/content/drive/MyDrive/DeepAA_stability_v1_D020`.
4. Run ZIP verification and restoration, pinned isolated environment installation,
   dataset verification/local copy and orchestration tests. Any failure stops.
5. Run readiness only: frozen membership/order, all dataset byte hashes, visible
   GPU, pinned versions, GPU placement, float32, two K7/seed113 training-only
   benchmark repeats, source/boot identity, available resources and a permanent
   Drive write/readback probe. No primary attempt is claimed by readiness.
6. Review READINESS.json and its hash, GPU/build/freeze/timing details and all
   logs. Readiness is not automatic launch permission. Target installation,
   Linux alias execution, Drive access and GPU checks must actually pass.
7. For ONE senior manual launch, set FROZEN_K7_LATENT6_CONFIRMED=True, paste the
   reviewed readiness digest, type the exact single-runtime confirmation, and
   set PRIMARY_LAUNCH_ENABLED=True in the final cell. Do not run that cell until
   those statements and target checks are satisfied. It invokes only
   `colab_k7_single.py launch --k 7 --latent-dimension 6 --seed 113`.

The CLI accepts only this identity. K5, latent4 and other seeds are refused;
the legacy six-run CLI remains disabled. The worker also restricts training to
K7/seed113. No model resume, second partial attempt or next-seed loop exists.

## Claims, persistence and provenance

Linux flock and process identities protect one runtime. The permanent
`attempts/K7_seed113` directory is created before spawning; its mere existence
blocks any future launch, even if empty, failed or completed. Existing local
K7/seed113 folders, QUEUE_ACTIVE.json, SINGLE_ACTIVE.json and unresolved durable
attempts also block. No stale cleanup, takeover or alternate project namespace.
Drive is NOT an atomic cross-runtime lock: the senior must disconnect/delete
previous project runtimes and use only one notebook session. The Windows CPU
run may continue independently under its separate ownership.

Registration binds K7/latent6/seed113, fresh epoch1, package/head/readiness hashes,
GPU provenance and explicit timestamped output/log paths. Worker process/exit
records and snapshots persist to Drive. After training, the unchanged validator
must pass, final snapshot hashes must match, and an archive must verify before
DURABLE_COMPLETE.json is written. Failures retain the claim and evidence.
Return single_run_result.json, REGISTRATION.json, DURABLE_COMPLETE.json,
validation report, archive hash and process/environment records. Nothing is
automatically combined with Windows results. Future evaluation must use these
explicit records and verified transfers, never a newest-directory search.

Local tests use fixtures for subprocess orchestration. Actual release restoration
is exercised through bundle checkout/overlay/hash/index checks on Windows; the
same regression executes full restore on Linux when available. Linux/Colab GPU
execution is pending and must not be described as tested locally. The notebook
always defaults to disabled primary launch.
