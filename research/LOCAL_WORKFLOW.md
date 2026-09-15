# Local stability_v1 workflow

This workflow completes the frozen matrix through scripts the senior developer launches. It runs independently of Codex, ChatGPT, credits or an open AI session. Only `finish` launches primary training. No new prompt/authorization is required between successful matrix runs. There is **no model resume support**.

## Entry points and source contract

Entry point: `research/tools/stability_workflow.py`.

- `status`: read-only inventory of ten run paths, eight comparison paths, Git status and lock. `COMPLETED_UNVERIFIED` is deliberately not a validity claim.
- `preflight`: pinned branch/HEAD/clean-worktree, environment, resources, dataset/manifests and partial-output checks. No training, TensorFlow model construction or output mutation. Full run/pair validation follows during finish/summarize.
- `finish`: sequentially validate existing work, run missing training, validate it, align/evaluate missing pairs with existing tools, verify comparisons, archive evidence and aggregate.
- `summarize`: validate available completed runs and comparisons, then make a new summary; never train or create missing comparisons. Missing work yields INCOMPLETE. Partial/corrupt work stops with an error rather than silently excluding it.
- `watch`: separate read-only viewer, automatically switches to current session events and newest training log. This exact absolute-script viewer is excluded from the competing-Python check.
- `update-docs`: explicit final step appending evidence-linked dated updates to project state, registry, Gate1 and stability overview. Never called automatically during the queue.
- `release-stale-lock`: explicit recovery action, checks recorded PID plus process creation identity and absence of competing processes; preserves lock metadata under released_locks. Never kills a process.

Implementation: `research/code/workflow.py` (orchestration), `research/code/workflow_evidence.py` (generalized validator, comparison verification, archive and aggregation). `_validate` and `_verify_pair` are read-only worker subcommands used internally, not training alternatives. Tests: `research/tests/test_workflow.py` (fixtures/stubs; no training).

Repository root is derived from the entry-point location, not the shell working directory. Default external state is `C:\Users\UserPc\Desktop\ai\stability_workflow_local`. The approved full HEAD is always explicit. Pin the reviewed NEW commit, not the previous `1a7214d...` preparation revision. The user handles all Git mutations.

The initial inspection found K5/113,211,307 completion markers, no active training, and the existing first-pair comparison. Seed307 was independently validated during build without training. K5/401,503 and all five K7 paths were missing. Existing old run revisions are explicitly allowed for their known run IDs; subsequent workflow-approved pins are recorded outside tracked source. Validation also checks unchanged scientific source Git objects, configuration and frozen hashes. Older valid runs are not rejected solely for having another execution revision.

## Frozen behavior and safety

Order: K5 113,211,307,401,503, then K7 113,211,307,401,503. Completed work is validated before skipping. No tuning or stopping on metric magnitude. Each missing run is a new Python process with its seed and existing deterministic controls. Architecture/objective/preprocessing/partial batches/50 epochs/final checkpoint are unchanged. Parent native exit codes are taken from Popen.wait; child output is redirected directly to exclusive log files, not shell pipelines.

The workflow uses an exclusive-create external lock plus read-only Windows process checks. Other Python/WSL/training processes block conservatively (except its own ancestor launchers and the exact read-only watch command). `.RUNNING`, FAILED, empty pre-created or other partial run directories are refused. Existing partial/stale analyses are refused, not silently completed or overwritten. Completed/nonempty originals are never deleted or changed. The existing runner's independent clean-worktree/output guards remain intact. Do not launch a second manual runner alongside the queue; an unrelated launcher cannot be forced to honor this workflow's lock, but process/marker checks run before each child phase.

Before/between phases: exact reviewed HEAD, intended branch, clean tracked/index/untracked source; protected source/manifests; Python3.13.3 and audited package versions; fresh Windows RAM/disk/process checks. Dataset applicability uses existing load_journal_v1: frozen counts/order/classes/disjointness and all image paths. Previous readability and CPU equivalence evidence are applicable; this does not re-decode every JPEG at each preflight. Later decoding errors still stop execution.

Engineering safety floors are 1 GiB available RAM and 2 GiB free disk; archives additionally require twice their source size plus 2 GiB. These are operational stop conditions, not scientific parameters or guarantees of peak memory. Keep stable power, adequate headroom and persistent storage. No automatic power changes, upgrades, downloads, uploads or process termination. If the resource check fails, resolve the actual resource problem manually without changing model settings.

## Validation, reuse and aggregation

Run validation generalizes the earlier seed113/211 read-only audit: exact prescribed configuration/provenance/seeds, all50 finite history rows and total arithmetic, inventory/hash/size/file-set checks, final checkpoint reload with30 finite tensors, decoded reconstruction from checkpoint, exact frozen mappings/labels, coefficient/latent shapes and simplex constraints, PNGs, nearest identities and class tables. Original hashes are checked before/after. This is artifact/state consistency, not a replay of the training trajectory or a scientific success threshold.

After a reference/candidate validate, existing validation-only Hungarian and held-out evaluation tools are used unchanged. Reference113 separately perK; cost1-Pearson, population std<=1e-8 fallback2*MAE; per-cell method matrix and explicit binary mask/permutation direction retained. Test never selects the permutation. Existing first-pair reuse checks input identities, rederives the validation assignment, compares every saved summary/table with existing metric functions, and verifies its prior derived SHA-256 ledger. All reuse requires numerical verification, not a directory marker alone.

New pairs are under `research/analysis/stability_v1/K{K}_113_vs_{seed}_workflow_v1/`; existing K5/211 remains `K5_113_vs_211_20260913/`. Aligned raw numbering remains separate from original outputs. Coefficients aggregate over748 lesion means; semantic weights over1002 images; nearest sets over10 test images; decoded RGB128 SSIM range1 and pixel MSE. Only lesion vectorL1 and dominant agreement receive the prescribed1000-draw seed2026 percentile95% bootstrap. Undefined correlations stay null/false (CSV: undefined); no replacement values.

Each summary is a new timestamped directory under `research/analysis/stability_v1/workflow_summaries/` with:

- `summary.json`: validated input provenance/hashes and all raw results.
- `raw_metrics.csv`: every per-pair/per-archetype coefficient, semantic, nearest and visual metric; pair-level lesion-bootstrap bounds.
- `descriptive_statistics.csv`: separate K5/K7 mean, sampleSD, min/max and expected/observed/defined/undefined denominators. Incomplete or undefined-value statistics are explicitly conditional descriptive summaries, not fabricated four-value results.
- `K5_seed*.svg` / `K7_seed*.svg`: labeled aligned side-by-side original PNGs.
- `FINDINGS.md`: technical findings and limitations; no stable/unstable threshold, pooledK conclusion or clinical claim.
- `summary_files_sha256.json`.

Only exact verified sets of all10 runs and8 pairs yield COMPLETE. Input hashes are rechecked before aggregation. Cross-run descriptive variation is kept separate from within-pair lesion-bootstrap uncertainty; shared-reference comparisons do not produce an inferential seed confidence interval.

## Evidence and locks

State root contents: `sessions/<UTC>/` (invocation, durable events, subprocess identity/native exit records, validation/pair reports, partial checkpoints, result); `training_logs/` (unique per-run flushed logs); `approved_pins/`; `archives/`; `released_locks/`; optional active `workflow.lock.json`.

Archives are content-addressed per immutable run/pair/summary. A run ZIP contains its artifacts, exact external training log and validation report (plus prior external validation evidence for113/211). A pair ZIP contains original separate derivatives, verification and mask/direction evidence. A summary ZIP includes findings/figures/tables plus that session's logs/receipts. ZIP CRC and per-member SHA-256 are checked; an existing archive is verified and reused, not overwritten. Archives do not recursively include previous archives. Dataset and full repository history are NOT backed up by these archives. Prior manually created same-disk archives remain untouched. New state/logs after an archived checkpoint may not yet be archived if a job fails.

**Same-disk archives do not protect against disk failure.** No uploads are performed. Keep archives and ignored analysis/run outputs independently of Git; committing documentation alone does not preserve them.

## Review, commit and pin — user commands only

HANDS does not execute these Git mutations. In Windows PowerShell, inspect then stage only the five explicit new files:

```powershell
Set-Location -LiteralPath "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000"
git status --short
Get-Content research\LOCAL_WORKFLOW.md
Get-Content research\code\workflow.py
Get-Content research\code\workflow_evidence.py
Get-Content research\tools\stability_workflow.py
Get-Content research\tests\test_workflow.py
git diff --check
git add -- research/LOCAL_WORKFLOW.md research/code/workflow.py research/code/workflow_evidence.py research/tools/stability_workflow.py research/tests/test_workflow.py
git diff --cached --check
git diff --cached --stat
git commit -m "research: add verified sequential stability matrix workflow"
```

After reviewing the successful commit, pin the actual resulting full HEAD. No unknown revision is hardcoded:

```powershell
Set-Location -LiteralPath "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000"
$approvedHead = & git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Cannot read HEAD." }
$sourceStatus = & git status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0 -or $sourceStatus) { throw "Source must be clean." }
New-Item -ItemType Directory -Path "C:\Users\UserPc\Desktop\ai\stability_workflow_local" -Force | Out-Null
$pinFile = "C:\Users\UserPc\Desktop\ai\stability_workflow_local\approved-head.txt"
if (Test-Path -LiteralPath $pinFile) { throw "Pin already exists: inspect it; do not silently replace it." }
New-Item -ItemType File -Path $pinFile -Value ($approvedHead.Trim() + "`n") | Out-Null
```

An existing `.pytest_cache` permission warning may remain because it is sandbox-owned. Nothing here alters permissions, suppresses Git stderr or weakens the cleanliness guard. Git ownership/access failures under a different account must be investigated, not bypassed.

## User-launched commands (any working directory)

Status, no pin needed:

```powershell
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" status
```

Preflight:

```powershell
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" preflight --approved-head ((Get-Content -Raw -LiteralPath "C:\Users\UserPc\Desktop\ai\stability_workflow_local\approved-head.txt").Trim())
$preflightExit = $LASTEXITCODE
Write-Host "Preflight exit: $preflightExit"
```

Finish the entire matrix (launch only after successful preflight and manual review):

```powershell
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" finish --approved-head ((Get-Content -Raw -LiteralPath "C:\Users\UserPc\Desktop\ai\stability_workflow_local\approved-head.txt").Trim())
$workflowExit = $LASTEXITCODE
Write-Host "Workflow exit: $workflowExit"
```

Live viewing in a separate PowerShell window:

```powershell
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" watch
```

The main window displays phase/run messages and file paths. Actual training batch/epoch output goes directly to its unique training log; watch follows it. Closing/Ctrl+C in the viewer does not interrupt training. Ctrl+C or closing the finish window may interrupt the child and leave a partial run/lock; avoid it during training. A process that survives the parent still blocks another launch. No automatic killing/restarting.

## Restart after a completed run, failure or Windows restart

Run status and inspect processes. If no lock/partial work remains, rerun the **same preflight and finish commands above** with the same approved pin. Every existing completion is revalidated before skipping. No new Brain prompt is needed. This is workflow continuation, not model resume.

If a stale lock remains, and its process identity is confirmed gone, explicitly release it without deleting its evidence:

```powershell
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" release-stale-lock --approved-head ((Get-Content -Raw -LiteralPath "C:\Users\UserPc\Desktop\ai\stability_workflow_local\approved-head.txt").Trim())
```

This refuses a live owner or other Python workload and renames metadata into released_locks. It does not make partial training resumable. For partial/failed/corrupt run or analysis directories, stop: inspect the unique log and session STOPPED record, preserve the attempt, and obtain a concrete recovery plan. This workflow intentionally has no delete, overwrite, retry-partial, rerun-sibling or weights-resume switch. Replacing a failed primary attempt is not silently performed. Resource issues may be resolved manually then preflight rerun, but any partial output still needs explicit recovery handling. Source changes midqueue stop subsequent phases; never repin automatically while training runs.

## Final summarization and documentation update

Rerun summary without training (also supports clearly labeled incomplete work when missing, not corrupt/partial):

```powershell
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" summarize --approved-head ((Get-Content -Raw -LiteralPath "C:\Users\UserPc\Desktop\ai\stability_workflow_local\approved-head.txt").Trim())
$summaryExit = $LASTEXITCODE
Write-Host "Summary exit: $summaryExit"
```

Inspect the reported summary and FINDINGS.md. Complete success requires exit0 from finish, status COMPLETE, all10 verified run identities, all8 verified pairs, perK four candidates, and verified archives. No metric threshold is implied.

Only after the queue stops and the summary is reviewed, append canonical documentation (this intentionally dirties source and must not be followed by another launch until reviewed/versioned):

```powershell
$latestSummary = Get-ChildItem -LiteralPath "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\analysis\stability_v1\workflow_summaries" -Directory | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "summary_files_sha256.json") } | Sort-Object Name | Select-Object -Last 1
if (-not $latestSummary) { throw "No completed summary output exists." }
& "C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe" -B -u "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000\research\tools\stability_workflow.py" update-docs --approved-head ((Get-Content -Raw -LiteralPath "C:\Users\UserPc\Desktop\ai\stability_workflow_local\approved-head.txt").Trim()) --summary-dir $latestSummary.FullName
$docsExit = $LASTEXITCODE
Write-Host "Documentation update exit: $docsExit"
Set-Location -LiteralPath "C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000"
git diff --check
git diff -- docs/project_state.md docs/experiments/experiment_registry.md docs/gates/gate_1_stability_infrastructure.md docs/experiments/archetype_stability.md
```

The update step verifies summary/run/pair hashes, takes the workflow lock, preserves historical entries and refuses documenting the same summary twice. It never stages/commits. Status/date annotations are engineering records, not manuscript writing or methodological amendments.

## D-019 recovery and undefined results — 2026-09-15

The senior developer relayed explicit BRAIN authorization to retain degenerate seed401, export undefined zero-mass semantics, and accept verified separate recovery. See D-019 and the dated protocol clarification; this was not an original preregistered undefined-semantics rule.

Seed401 recovery is complete at `C:\Users\UserPc\Desktop\ai\stability_workflow_local\recovered_exports\K5_seed401_20260915`. Its `RECOVERED_EXPORT.json` records original training HEAD, recovery HEAD plus exact uncommitted code hashes and a source snapshot, original evidence hashes, zero optimizer steps and full verification. The original `research/runs/stability_v1/K5/seed_401/FAILED.json` and other original files remain unchanged. Missing historical runtime metadata is explicitly unavailable; recovery environment/timing is separately labeled. No ordinary COMPLETED marker is manufactured.

External `recoveries/K5_seed401.json` explicitly registers the receipt SHA-256. Workflow path resolution verifies registration, receipt, original/checkpoint identity, original/copied/exported hashes and code hashes; the normal full validator still runs before skipping. Only this authorized attempt receives this exception. Unregistered partial, active, failed or tampered attempts still block. A later change to recovery code requires explicit re-audit, not automatic acceptance. `status` reports RECOVERED_UNVERIFIED to distinguish an inventory view from full numerical validation.

Future runs export zero-mass distributions as JSON null/blank numeric CSV fields with status/reason and measured mass. JS is undefined if either matched distribution is undefined. Pair summaries report defined/K and means labeled "defined pairs only"; no valid matches means null. Aggregate CSV null numeric cells are blank (superseding this guide's older `undefined` text convention), with defined/undefined denominators and degeneration counts in summaries. All ten identities remain in the registered matrix view. D-016, null correlations, all other defined metrics, and bootstrap scope are unchanged. Degenerate usage cannot be inferred useful from distinct decoded images.

Verification/report and exact guarded review/commit/pin/preflight/continuation commands: `C:\Users\UserPc\Desktop\ai\seed401_recovery_20260915\REPORT.md`. Fifty fixture tests passed; real checkpoint-only recovery, full validation, first-pair regression, seed401 validation-only alignment/test evaluation, and an explicitly incomplete verification summary passed. The seed401 comparison is at `research/analysis/stability_v1/K5_113_vs_401_workflow_v1/`; the workflow will verify/reuse it rather than overwrite it. The verification summary is not a completed matrix study.

Do not run `recover_seed401_exports.py` again against the existing directory: it correctly refuses overwrite or duplicate registration. Recovery has already happened; the next user action is review/version the source and explicitly replace the old approved pin while preserving its original contents in a new backup file. Then use preflight and finish. **Continuing the queue is not resuming model training:** validated seed401 is skipped, and the next missing primary run is K5/seed503, followed by the five K7 seeds. No optimizer state/resume support was added. The queue's clean-source, process, resource and primary runner guards remain in force.
