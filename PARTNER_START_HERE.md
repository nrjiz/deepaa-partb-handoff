# DeepAA partner setup

This branch contains the original HAM10000 workflow and separate proposed JAFFE
and HAM10000 K3 experiments. Start here before the historical root README.
Nothing in this guide authorizes training. Real JAFFE training and a clean
installation on your machine remain to be verified.

## Get the code the first time

The intended repository below returned 404 during preparation on 19 October
2026. The owner must confirm its private visibility, grant access, and publish
the branch before these GitHub commands will work. Use the actual confirmed
repository URL if the destination changes. Never make the repository public to
work around an access error.

Install Git and Python 3.13.3, then open PowerShell in a directory for source code:

```powershell
git clone --branch jaffe-partb --single-branch https://github.com/Amy007-ali/Deep-Archetypal-Analysis-on-HAM10000.git deepaa-partb
Set-Location deepaa-partb
git branch --show-current
git status --short
```

The branch must be `jaffe-partb`. Use GitHub's normal sign-in flow; do not put a
password or token in the URL. A 404 can mean missing access or an unavailable
repository; ask the owner to resolve it.

For a private offline handoff, download `jaffe-partb-partner.bundle` and its
verification receipt together. Check the bundle SHA256 against the receipt:

```powershell
Get-FileHash .\jaffe-partb-partner.bundle -Algorithm SHA256
git clone --branch jaffe-partb .\jaffe-partb-partner.bundle deepaa-partb
Set-Location deepaa-partb
```

A bundle clone has a local-file origin. Once GitHub is available, point that
clone at the owner-confirmed private URL before fetching or pulling. The bundle
retains inherited source history, images, weights and author metadata; share it
privately. It contains no JAFFE images or local environment.

## Install and inspect without training

```powershell
py -3.13 -m venv .venv-jaffe
$py = (Resolve-Path .venv-jaffe/Scripts/python.exe).Path
& $py -c "import sys; assert sys.version_info[:3] == (3,13,3), sys.version"
& $py -m pip install -r jaffe_partb/requirements.txt
& $py -m pip check
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:MPLBACKEND = 'Agg'
& $py -B -m jaffe_partb plan
& $py -B -m jaffe_partb plan --config jaffe_partb/configs/jaffe_k3_proposed.json
& $py -B -m jaffe_partb plan --config jaffe_partb/configs/ham10000_k3_proposed.json
& $py -B -m pytest jaffe_partb/tests -q -p no:cacheprovider
```

These plans do not train, read real images, or authorize attempts. Tests use
engineering fixtures and some model inference, not real-data training or proof
of scientific stability. Exact dependency pins reflect the previously checked
Windows CPU environment. Do not silently replace them if installation fails;
record the failure and obtain a reviewed environment change.

Linux/macOS: use `python3.13 -m venv .venv-jaffe` and
`.venv-jaffe/bin/python` instead. Other platforms and GPU execution need their
own installation checks; Windows instructions do not establish GPU readiness.

## Choose the experiment

| Configuration | Dataset | K and latent dimension | Seeds | Reference |
|---|---|---|---|---|
| `jaffe_partb/proposal.json` | JAFFE | 5/4 and 7/6 | 113, 211, 307, 401, 503 | 113 |
| `jaffe_partb/configs/jaffe_k3_proposed.json` | JAFFE | 3/2 | 113, 211 | 113 |
| `jaffe_partb/configs/ham10000_k3_proposed.json` | HAM10000 | 3/2 | 113, 211 | 113 |

These are proposals, not accepted experiment authorizations. For different seeds
or K, copy to a new configuration with a new experiment ID and explicitly state
the full seed list and reference before inspecting outcomes. Do not edit the
completed HAM10000 `archetype_stability_v1` files. See
[new experiments](jaffe_partb/NEW_EXPERIMENTS.md).

## Before any real run

1. Read [the execution guide](jaffe_partb/README.md),
   [proposed protocol](jaffe_partb/PROTOCOL_PROPOSED.md),
   [dependencies](jaffe_partb/DEPENDENCIES.md), and
   [known checks and limits](jaffe_partb/CHECKS.md).
2. Obtain dataset images separately. For JAFFE follow the official record linked
   in the execution guide and its terms. No dataset is downloaded by this setup.
3. Use data and experiment workspaces outside the source checkout. Do not reuse
   historical run directories. Audit actual image bytes, labels and split coverage.
4. Have the scientific reviewer accept the exact configuration, source commit,
   data-manifest hash and requested attempts in a separate approval file. Decide
   JAFFE split/evaluation rules prospectively. HAM10000 approvals do not transfer.
5. Only then follow the single-run, validation, pair-verification and aggregation
   commands. Preserve collapsed runs, failures, nulls, denominators and claims.

JAFFE uses expression-label semantics and proposed participant-aware uncertainty;
HAM10000 uses diagnoses and its documented lesion rules. Participant IDs must not
be passed to the lesion evaluator. Matching uses validation only; test data
evaluates the fixed match. K−1 is the latent dimension, not the seven-class count.

## Receive later code updates

Finish or preserve local work first. Keep a record of the source commit used for
each approved experiment. Do not update code during an active approved run.

```powershell
git status --short
git switch jaffe-partb
git pull --ff-only origin jaffe-partb
git rev-parse HEAD
```

Stop if Git reports local conflicts or diverged history; do not reset, force-pull,
or overwrite experiments. A new source commit requires review before a new
source-bound training approval. Datasets and research outputs are not committed.

## Share results back

Keep the configuration, approval, image manifest, source commit, environment,
run logs, weights, raw exports, validation receipts, fixed matches, full pair
metrics and aggregation together outside Git. Preserve interrupted attempts and
undefined values. Use approved private storage and verify an independent copy;
another folder on the same disk is not a backup. Do not post JAFFE imagery in
GitHub issues, README files, or public result packages.
