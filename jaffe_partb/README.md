# JAFFE Part B / RQ1 handoff

For cloning, installation, configuration choice and later updates, start with
[the partner setup guide](../PARTNER_START_HERE.md).

This is a **separate proposed experiment**, not an amendment to completed HAM10000
`archetype_stability_v1`. Start here rather than the historical root README. No
JAFFE model has been trained or evaluated by this handoff. No HAM10000 outputs
are regenerated. Read `PROTOCOL_PROPOSED.md`, `HAM10000_PROVENANCE.md`,
`DEPENDENCIES.md`, and `CHECKS.md` before execution.

## Acquire the data yourself

Use the [official JAFFE v2 record](https://zenodo.org/records/14974867) and its
[README](https://zenodo.org/records/14974867/files/README_FIRST.txt?download=1).
Accept its terms and read the two required publications. The record restricts use
to non-commercial scientific research, prohibits redistribution and web posting,
and permits at most ten sample images in scientific publications. Do not commit
TIFFs, exports of real images, ZIPs, or results to this branch. Review permissions
before sharing face-derived visualizations or weights; permission is not inferred
from access to this repository.

The official metadata describes 8-bit grayscale TIFFs, participant codes and
requested-expression labels. These are posed expressions, **not measured internal
emotions**. The adapter uses the 213 actual archive filenames, not every row in
the larger ratings table. It never repairs label discrepancies by guessing.
`official_filename_inventory.json` contains public filenames only, no image bytes.

Extract locally **outside** the checkout. Keep the official downloaded ZIP and its
download provenance. The record lists ZIP MD5 `fe13f3302eb9968ef04367456f665436`;
check this against the actual download. Local `audit-data` additionally records
SHA256 for each image. Filename agreement alone cannot authenticate image bytes.

## Install and inspect (Windows PowerShell)

Obtain the `jaffe-partb` branch from the private repository
`https://github.com/nrjiz/deepaa-partb-handoff`. Ali Yassine (`Amy007-ali`) must
accept the owner's collaborator invitation before cloning. The older
`Amy007-ali/Deep-Archetypal-Analysis-on-HAM10000` destination is superseded.
From the checkout root, use Python **3.13.3**:

```powershell
py -3.13 -m venv .venv-jaffe
$py = (Resolve-Path .venv-jaffe/Scripts/python.exe).Path
& $py -m pip install -r jaffe_partb/requirements.txt
& $py -m pip check
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:MPLBACKEND = 'Agg'
& $py -B -m jaffe_partb plan
& $py -B -m pytest jaffe_partb/tests -q -p no:cacheprovider
```

`plan` is a dry run: prints settings, coverage, proposal hash and no-training
status. It does not create a claim, import TensorFlow, or inspect real images.
Check `python --version` is exactly 3.13.3. Pins reflect the existing tested local
environment; a clean download/install and your platform still need verification.
Do not silently change pins when installation fails. Record the resolved freeze.

On Linux/macOS use `python3.13 -m venv .venv-jaffe` and
`.venv-jaffe/bin/python`; the Python CLI is portable, but only Windows CPU was
checked here. No GPU readiness or cross-hardware numerical equivalence is claimed.

## Freeze the proposed experiment before training

The proposed deterministic partition is train KA/KL/KM/KR/MK/NA (129 images),
validation NM/TM (41), test UY/YM (43). All expressions occur in every partition.
It is an alphabetical, outcome-independent 6/2/2 participant allocation, not a
claimed optimal split. Decide whether this limited fixed-split generalization
question is the desired study before approval. A cross-validation study requires
a separately specified version; do not mix it into this one retrospectively.

```powershell
$data = 'D:\private-data\jaffe'       # actual extracted directory
$workspace = 'D:\private-research'
$work = "$workspace\jaffe_partb_v1"
New-Item -ItemType Directory -Path $work  # new external directory
& $py -B -m jaffe_partb audit-data --data $data --output "$work\MANIFEST.json"
Get-FileHash jaffe_partb/proposal.json -Algorithm SHA256
Get-FileHash "$work\MANIFEST.json" -Algorithm SHA256
git rev-parse HEAD
Copy-Item jaffe_partb/approval.template.json "$work\APPROVAL.json"
```

Review image format, coverage and hashes. The adapter refuses missing/extra
images, duplicate names, participant leakage, missing classes, unexpected TIFF
format and changed bytes. It does not fetch or redistribute images. Actual
decoding/dimensions must pass on your copy; the official filename audit is not
an image audit.

An actual scientific reviewer must accept the proposal and fill the approval:
`status=ACCEPTED`, reviewer, UTC time, exact config/manifest SHA256, **current clean
source HEAD**, terms acceptance, evaluation-limitations acceptance, and explicit
`authorized_attempts`, e.g. `[[5,113]]`. The template grants nothing. Accepted
HAM10000 D-022/D-023 decisions do not authorize JAFFE. Record proposed changes in
a new version and review them before running; do not edit the frozen HAM files.

## Execute exactly one approved run (not executed by this handoff)

```powershell
$env:PYTHONHASHSEED = '113'
$env:TF_DETERMINISTIC_OPS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
& $py -B -u -m jaffe_partb train --k 5 --seed 113 --data $data --manifest "$work\MANIFEST.json" --approval "$work\APPROVAL.json" --workspace $workspace
```

Each invocation starts a fresh process and fresh original-stochastic model, runs
50 epochs, and exports final weights. Never load HAM10000 or pilot weights. No
resume, retry or queue exists. Run only one process using this workspace; use one
canonical workspace for all attempts. The OS lock is local, not a distributed
lock. Permanent `claims/K5_seed113.json` and `runs/K5_seed113` are never reused.
Failure leaves the claim and partial run. An interruption is not permission to
change workspace to bypass the claim. Preserve it and review a new-attempt
exception separately. Close other resource-intensive work manually as needed;
the launcher does not terminate applications or manage system power.

Monitor in a second PowerShell window:
```powershell
Get-Content "$work\runs\K5_seed113\training.log" -Tail 20 -Wait
```

Run each other approved K/seed manually with matching `PYTHONHASHSEED`. Reference
is always 113 within K. Keep collapsed and unfavorable results. K5 and K7 change
latent dimension along with K; do not attribute differences to K alone.

## Independently validate, align, evaluate and preserve

Use a **new process** for each command; receipt/output names must be absent:

```powershell
& $py -B -m jaffe_partb validate --run "$work\runs\K5_seed113" --data $data --receipt "$work\K5_seed113_VALIDATION.json"
# After independently training and validating seed211 with its own approval:
& $py -B -m jaffe_partb pair --reference "$work\runs\K5_seed113" --candidate "$work\runs\K5_seed211" --reference-receipt "$work\K5_seed113_VALIDATION.json" --candidate-receipt "$work\K5_seed211_VALIDATION.json" --output "$work\K5_113_vs_211"
& $py -B -m jaffe_partb verify-pair --pair "$work\K5_113_vs_211" --receipt "$work\K5_113_vs_211_VALIDATION.json"
& $py -B -m jaffe_partb figures --pair "$work\K5_113_vs_211" --receipt "$work\K5_113_vs_211_VALIDATION.json" --output "$work\K5_113_vs_211_figures"
& $py -B -m jaffe_partb preserve --source "$work\runs\K5_seed113" --archive "$work\K5_seed113.zip"
```

Run validation checks hash inventories, all 50 epoch records, final tensor hashes,
fresh-model weights reload, saved outputs, metrics, decoded PNGs and nearest
references. Replay tolerance is explicitly `rtol=1e-6, atol=1e-7`, a **new JAFFE
engineering proposal**, not a retrospective relaxation of HAM validation. It does
not replay optimizer history or prove that every training operation occurred.

Pair verification independently recomputes saved-data calculations in a new
process. Matching sees only validation. Raw arrays, aligned derivatives, cost
method mask, costs and the zero-based saved permutation are in
`alignment_and_outputs.npz`. Nulls, zero-mass reasons, denominators, assignment
ambiguity, expression JS, nearest-real Jaccard/expression JS, SSIM and MSE are in
`summary.json`. No test-data tie breaking is performed.

For the complete eight-pair aggregate:
```powershell
$pairs = @()
$receipts = @()
foreach ($k in @(5,7)) {
  foreach ($seed in @(211,307,401,503)) {
    $pairs += "$work\K${k}_113_vs_${seed}"
    $receipts += "$work\K${k}_113_vs_${seed}_VALIDATION.json"
  }
}
& $py -B -m jaffe_partb aggregate --pair $pairs --receipt $receipts --output "$work\aggregate"
```

This loop only assembles paths; it never launches training. Means/SD use all four
pairs per K; there is no seed-level inferential CI. Undefined semantic/column
values remain in the full pair records rather than being imputed or filtered.

Archive receipts verify ZIP readback **only**. Copy run, receipt, claims, approval,
manifest, source commit/bundle, pair packages and receipts to approved independent
storage; retrieve and hash-check them. Another folder on the same disk is not
independent protection. Keep a custody record; no automatic cloud backup exists.
Use `git bundle create source.bundle HEAD` outside the checkout for source history.
Do not commit generated results or JAFFE imagery. A local PASS is not a scientific
finding, independent backup, or a license to publish identifiable face images.

## Additional K values, seeds and HAM10000

See [NEW_EXPERIMENTS.md](NEW_EXPERIMENTS.md). Named new experiments support explicit K2?10 and explicitly listed seeds/reference. K3 examples exist for both datasets. The original HAM runner, manifests and accepted outputs remain unchanged.
