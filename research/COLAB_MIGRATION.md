> D-021 supersedes the six-run procedure below. For this release use [COLAB_K7_SINGLE.md](COLAB_K7_SINGLE.md): one independent K7/latent6/seed113 run. The legacy queue stays disabled. Historical notes follow.

# D-020 supervised Colab migration

Implemented 2026-09-21 following the senior's prospective instruction. This
supersedes the September 5 provider hold and the proposal to stop before K7.
No further Windows primary training. GPU compatibility, timing and actual
readiness remain **PENDING until Colab execution**.

Open `research/notebooks/stability_v1_colab.ipynb`. Setup never launches primary
training. The final cell defaults to disabled and requires a reviewed readiness
SHA-256 and the exact single-runtime declaration. Do not use `aanet_ham10000 (2).py`
or `aanet_outputs`. The Drive HAM10000 paths are candidates, not verified inputs.

## Scientific and operational scope

One new K5/latent4/seed503 attempt begins at epoch 1. Its predecessor stays
interrupted, in place and hash-bound in a separate preservation copy. Then K7
seeds113,211,307,401,503 execute sequentially. No automatic second attempt.
Architecture, float32, RGB128, batch32, epochs50, Adam0.001, losses1/1/.01/1/0,
frozen manifests, seed matrix, final-checkpoint selection and evaluation rules
are unchanged. D-019 remains specific to seed401. Completed work is validated,
never retrained. Mixed CPU/GPU and environment differences limit a seed-only
interpretation; no CPU/GPU numerical-equivalence claim is made.

## Packaging and provenance

Setup fix (2026-09-22): immediately after importing `sys`/`os`, the first setup
cell sets `sys.dont_write_bytecode = True` and
`os.environ['PYTHONDONTWRITEBYTECODE'] = '1'`, before any research package import.
The first setting protects the running notebook kernel; the environment setting
also protects child interpreters. Without them, importing `colab_transfer` can
create `__pycache__/*.pyc` inside the extracted payload before `restore()` checks
its inventory. Those additions correctly fail the unchanged integrity check.
The fresh-process regression starts with bytecode enabled, executes the notebook's
actual bootstrap settings, imports extracted transfer code and verifies that
the complete file/directory inventory remains unchanged.

Use the newly committed/rebuilt release notebook and ZIP together. For the
reported setup failure (WORKSPACE absent, payload containing new bytecode),
start a fresh runtime and rerun the corrected setup from a fresh extraction.
Do not amend PACKAGE.json, exclude bytecode from inventory verification, or
overwrite a previous release/preservation package. Original research evidence
and all integrity checks remain unchanged.

`build_colab_transfer.py` writes only a new external destination. It checks free
space, copies/hashes the interrupted run, log, both relevant sessions, lock and
pin, inventories all 10015 JPEGs and metadata without copying images, and includes
a complete Git bundle plus exact reviewed source bytes, manifests, K5 runs,
existing pairs, D-019 receipt/registration/source snapshots/archive and evidence.
No environment, credential store, dataset archive or Git remote config is copied.
Original evidence is rehashed after packaging. ZIP members are SHA-256 verified.

A build without `--approved-head` is a **disabled draft** exposing uncommitted
status. A release build requires an exact explicitly supplied clean HEAD; it
does not modify the Windows approved pin. The new package HEAD is the Colab
execution approval; the old Windows pin is retained as historical evidence.

Restore uses an isolated clone of the bundle, LF checkout (`core.autocrlf=input`,
`core.eol=lf`) with a disposable `.git/info/attributes` rule giving only the
protected baseline `text eol=crlf`, and exact packaged working-tree bytes.
It verifies Git cleanliness, every index mode/blob/stage/path against the reviewed
commit before and after overlay, all source-byte
hashes and the existing literal protected-source/manifests hashes. No accepted
hash is substituted. D-019-bound files remain unchanged.

Restore fix (2026-09-22): the actual d6adf95 release reproduces the reported
false-dirty checkout on Git for Windows 2.46.0. Blanket `core.autocrlf=true`
checks out CRLF, then the snapshot restores LF in docs/research files. Their
normalized blob IDs match HEAD, but cached checkout sizes differ. The content
diff is empty while status reports modifications; `update-index --refresh`
still reports `needs update`. Matching the initial checkout line endings to the
snapshot fixes the failure without staging content, resetting files, changing
index entries, ignoring paths or rewriting packaged evidence. The protected
baseline remains byte-for-byte CRLF. The exact snapshot and clean-status gates
still reject any incompatible future package.

The regression extracts the actual release ZIP, checks the receipt and full
inventory, reproduces the old failure, tests the corrected checkout and all
source hashes, and verifies that a real edit remains blocked. It runs full
restoration including historical aliases and unchanged D-019 registration,
receipt and archive verification when Linux is available. Set
`DEEPAA_TEST_RELEASE` to a release directory containing its ZIP and receipt to
test another release. On Windows only checkout/overlay verification executes;
this is not a Linux or GPU success claim.

`RELOCATION.json` separately binds old paths to packaged files. On Linux, aliases
for exact historical path strings live in an external worker working directory,
never in the checkout. Original receipts and pair paths are not edited and
unchanged verifiers still resolve and hash the actual bytes. Target readiness
must successfully verify the real seed401 receipt and existing pairs. Any alias,
path, numerical or environment failure blocks training, rather than weakening
verification. The alias scheme is Linux-specific and needs target confirmation.

## Environment and readiness

Setup now reports restore phases and dataset-copy progress. Installation and
test subprocess output streams into the notebook. Readiness keeps visible
progress/errors in `readiness_stdout.txt` and parses only the separate
`readiness_result.json`. Worker start/exit messages identify each log; failures
also print its last 12 KB. Nonzero exits stop the cell. `restore_d019.log` and
`RESTORE.json` must confirm the real registration/receipt/archive gate before
installing the target environment. This early gate uses no TensorFlow/model.

Local verification on 2026-09-22 had no Linux runtime (WSL not installed; Docker
unavailable). If the failed Colab runtime is still present, this optional single
read-only cell captures the remaining Linux-specific index/attribute evidence;
it does not repeat the supplied status/diff diagnostics or modify the index:

```python
import sys, os
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import subprocess
r = Path('/content/deepaa_workspace/repo')
def inspect_git(*args):
    result = subprocess.run(['git', '--no-optional-locks', *args], cwd=r,
                            capture_output=True, text=True)
    print(args, 'exit=', result.returncode, result.stdout, result.stderr)
inspect_git('--version')
inspect_git('config', '--show-origin', '--get-regexp', r'^(core\.(autocrlf|eol|checkstat)|diff\.autoRefreshIndex)$')
for name in ('docs/README.md', 'daa_ham10000_interpretability_fixed_v2.py'):
    print(name, 'actual_size=', (r/name).stat().st_size)
    inspect_git('ls-files', '--debug', '--', name)
    inspect_git('check-attr', 'text', 'eol', '--', name)
    inspect_git('hash-object', '--path='+name, name)
    inspect_git('rev-parse', 'HEAD:'+name)
```

After senior review/commit/rebuild, use the new notebook, ZIP and receipt as one
release. Preserve the failed extraction/checkout; do not repair it in place.
To use the corrected release in the same supervised runtime, set `PAYLOAD` and
`WORKSPACE` to new absent `/content/...` paths in the setup cell before running
verification. The reported failed restore never installed `ENV` or prepared
`LOCAL_DATA`; keep their existing absence checks. Do not change the fixed
`DURABLE` project location or clear any attempt/launch records. Run setup through
readiness in order and stop for senior review with primary launch disabled.

Use isolated Python3.13.3 and `requirements_colab.txt`, not notebook kernel
defaults. Missing wheels, driver incompatibility or failed installation stop the
procedure; do not substitute versions. Installation records, full freeze and
wheel hashes from pip's report provide a target-specific reproduction lock.
Review them before approving readiness; the Linux resolution is not claimed
verified on Windows. Record CUDA/cuDNN build information and actual GPU/driver.

Readiness hashes all Drive images and metadata, then verified local copies. It
checks two independent repeats of a labeled training-only benchmark for each K:
one frozen training batch, 3 warmup +20 measured steps, no held-out selection.
Exact repeated weights/loss fingerprints and deterministic-op support are
required. GPU placement/memory and training-only time estimates are reported;
these exclude validation/export/persistence and do not promise runtime lifetime.
Historical inference validation is CPU-only and uses zero optimizer steps;
all primary training requires a visible GPU. Any historical validation failure
requires review, not changed tolerances.

## Single-runtime safety accepted by the senior

Drive records are **not an atomic cross-runtime lock**. Before primary launch,
the user must disconnect/delete every previous project runtime and use one
project notebook session only. The launch cell requires this exact declaration:

> I deleted or disconnected all previous project runtimes and will use only this project session.

Linux `flock` and /proc boot-ID/start-tick process inspection protect against
local duplicate execution. The same fixed durable project directory must be
used in all sessions. `QUEUE_ACTIVE.json` and unresolved attempt registrations
block subsequent execution. Never delete them to make a cell pass. There is no
automatic stale-record cleanup, timeout, takeover, process kill or retry.
After an interrupted runtime, senior review must establish its termination and
decide any record transition; no transition tool is supplied here.

## Durability and limits

Every roughly two-second poll snapshots stable-readable artifact bytes and logs
into immutable, hash-named Drive objects and dated snapshot manifests. A file
can be copied during a pause in writing: partial snapshots are explicitly not
completion or resumable checkpoints. Final weights are copied as soon as
observed, with final full hash/validation verification before the next run.
Abrupt termination can still lose the last unpersisted bytes or interrupt a
Drive write. Mounted Drive readback checks do not guarantee remote flush timing.

After each successful run the entire artifact set is validated, readback-hashed,
archived and sealed with `DURABLE_COMPLETE.json`. Later sessions require that
receipt, exact registration/package, all object hashes and full validation
before skipping. A missing receipt means unresolved, even if weights exist.
If persistence fails, the supervisor records failure and never starts another
run. It does not kill an existing child; it continues supervising its exit.
Workflow continuation is **not model-training resume**. No intermediate model
checkpoint, altered selection rule or keep-alive is introduced.

Numerical summaries count each K/seed once. A separate hashed summary bundle
retains both seed503 attempts. No manuscript output is generated.

## Senior sequence

1. Review new files, D-020 and local fixture evidence. Stage/commit manually.
2. Review the resulting full HEAD; build a new release package with that HEAD.
   Do not change the old Windows lock, marker or approved pin.
3. Upload the release ZIP and receipt to `MyDrive/DeepAA_migration`; open the
   notebook in Colab, choose GPU and run setup/readiness only.
4. Paste back READINESS.json plus its SHA-256, package receipt, GPU/environment,
   benchmark timing/memory, deterministic-repeat result and any failed cell.
5. Only after review, manually enable the final cell and supply the exact
   declaration and reviewed readiness hash. No additional K/seed settings exist.

Do not reuse a draft ZIP after committing: rebuild to record real committed
provenance. Do not execute historical launch examples from older documents.

Provider references: https://www.tensorflow.org/install/pip and
https://research.google.com/colaboratory/faq.html. GPU availability and runtime
limits vary; the notebook does not guarantee completion in one session.
