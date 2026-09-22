# D-021: one manual Windows seed503 attempt

Authorized prospectively 2026-09-22 by the senior's explicit split-execution
direction. K5/latent4/seed503 belongs exclusively to Windows; K7 belongs to Colab.
This does not extend D-019, authorize a matrix queue, or change scientific settings.

Entry point: `research/tools/local_seed503.py`. Fixed state is the existing
external `stability_workflow_local`; no alternative state directory or identity
flags can allocate a second attempt. `inspect` is read-only and requires process,
original marker/hash, Python/package, memory/disk, frozen split and JPEG hash
checks. `preflight` also requires the explicitly approved clean HEAD and no lock.
It is expected to block until the senior commits and handles the existing lock.

Senior sequence: review/commit these changes; inspect with that exact HEAD and
the released `payload/DATASET.json`; register BEFORE releasing the historical
lock; then use existing `stability_workflow.py release-stale-lock` with the same
approved HEAD. Registration requires the original lock's exact SHA-256 and dead
owner identity. It preserves that lock, original run, both sessions, training log
and old pin before release. It writes one exclusive external `local_seed503`
folder with a registration, preservation manifest, dataset inventory and environment
record. An incomplete registration blocks another registration; do not delete it.

Then rerun preflight and manually invoke `launch` with the exact ownership
confirmation printed in the chat instructions. Registration explicitly maps
K5/seed503 to `K5/seed_503_attempt_<UTC timestamp>` and a dedicated external log.
The original remains interrupted and in place. The launch takes the unchanged
workflow lock, creates a permanent LAUNCH.json claim BEFORE spawning, and invokes
the unchanged `run_stability_one.py` once with k5/seed503, no rerun/resume flags,
fresh initialization from epoch 1, frozen manifests and CPU execution. It records
child identities and exit status. No second launch is supported, including an
exit before output creation. No K7 path or full finish queue is invoked.

On success the unchanged validator runs in a fresh CPU interpreter and writes
VALIDATION.json. A failed/interrupted validation never triggers training again.
Manual `validate` may be used only after successful child exit and before a
validation receipt exists. No completion is inferred from a marker alone.
The unchanged global workflow still sees the original marker and stays blocked;
it must not be used to finish, compare or aggregate this attempt. Future combined
analysis must explicitly consume this registration and validated provenance,
never choose the latest directory. Archive/verified transfer and K7 integration
are separate work, not required to start this single local run.

Only the senior may stage, commit, register, release the lock or launch. Existing
Python processes (including unrelated Streamlit) remain blockers; manually stop
them through their normal UI if appropriate. A missing old PID does not suffice:
registration, stale release and launch each obtain fresh process checks.

Before local launch, confirm no Colab project training is running. Existing Colab
release notebooks predate D-021 and must not be launched. Continue Colab development
in an isolated checkout once local training starts; never edit the active checkout.
Windows approved-head.txt stays historical and unchanged; explicit CLI HEAD plus
the registration bind this one execution. No numerical tolerances, optimizer,
epochs, splits, matching, selection or evaluation rules are changed.
