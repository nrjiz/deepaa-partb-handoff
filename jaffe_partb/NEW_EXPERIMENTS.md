# Preserve HAM10000; configure separate experiments

The CLI package name `jaffe_partb` identifies this handoff, but the new-experiment
adapter supports **JAFFE and HAM10000**. The original `research/tools/` HAM
training/alignment/evaluation commands remain intact. Their frozen K5/K7
configuration, seeds, `journal_v1`, completed artifacts and authorization history
must not be edited to accommodate new K values.

Examples, both **PROPOSED**:

| File | ID | Dataset | K / latent | Seeds | Reference | Evaluation |
|---|---|---|---|---|---|---|
| `proposal.json` | `jaffe_partb_v1` | JAFFE |5/4,7/6|113,211,307,401,503|113|image metrics + participant-cluster uncertainty|
| `configs/jaffe_k3_proposed.json` | `jaffe_k3_example_v1` | JAFFE |3/2|113,211|113|same proposed JAFFE rules|
| `configs/ham10000_k3_proposed.json` | `ham10000_k3_example_v1` | HAM10000 |3/2|113,211|113|frozen lesion aggregation/bootstrap and diagnosis semantics|

Each configuration lists explicit K values (2–10), seeds and reference, epochs,
batch, optimizer, preprocessing, labels, split and evaluation rules. The K limit
keeps exhaustive assignment-ambiguity diagnostics bounded; larger K requires
separate implementation review. Latent dimension is always K−1, not a free hidden
parameter. The classifier still has seven dataset-specific classes. Seeds must
be distinct nonnegative integers and contain the predetermined reference. A new
seed list is not inferred from earlier approvals. At least two seeds are required
for a stability comparison. With one pair, sample SD is null with a reason.

Clone a proposal to a new file, choose a new `experiment_id`, specify the full
list and reference **before looking at outcomes**, commit and review it, then
audit data and approve its hashes. Never reuse an existing ID for different
settings. Runtime `EXPERIMENT.json` permanently binds ID/config/data/source; each
run has a separate permanent claim and non-overwriting directory. Claims never
authorize retries or finding a more favorable reference. Use one canonical
workspace, not several workspaces to bypass claims.

```powershell
# Safe plan only: no TensorFlow model, images or claims are created.
& $py -B -m jaffe_partb plan --config jaffe_partb/configs/ham10000_k3_proposed.json
& $py -B -m jaffe_partb plan --config jaffe_partb/configs/jaffe_k3_proposed.json
```

For a **new HAM K3 experiment**, the existing HAM images remain external and
unchanged. `--data` is the directory containing both
`HAM10000_images_part_1`, `HAM10000_images_part_2`, and
`HAM10000_metadata.csv`. The adapter calls the original frozen manifest loader
and TensorFlow JPEG pipeline. It hashes all 10,015 images and metadata into a
**new** manifest; no original manifest is regenerated. The reviewer must bind
these bytes to the existing verified dataset inventory when reusing local data.

```powershell
$cfg = 'jaffe_partb/configs/ham10000_k3_proposed.json'
$data = 'D:\private-data\HAM10000'
$workspace = 'D:\private-research'
$work = "$workspace\ham10000_k3_example_v1"
New-Item -ItemType Directory -Path $work
& $py -B -m jaffe_partb audit-data --config $cfg --data $data --output "$work\MANIFEST.json"
# Review proposal/source/data and fill a separate APPROVAL.json first.
# The following is an example for an approved attempt, not authorization:
$env:PYTHONHASHSEED='113'; $env:TF_DETERMINISTIC_OPS='1'; $env:CUDA_VISIBLE_DEVICES='-1'
& $py -B -u -m jaffe_partb train --config $cfg --k 3 --seed 113 --data $data --manifest "$work\MANIFEST.json" --approval "$work\APPROVAL.json" --workspace $workspace
& $py -B -m jaffe_partb validate --config $cfg --run "$work\runs\K3_seed113" --data $data --receipt "$work\K3_seed113_VALIDATION.json"
```

Use the same pattern for JAFFE K3 with its configuration and external TIFFs.
Pair commands take the approved protocol from the two runs and refuse mixed
K/configuration/data/source or a different reference. Aggregation requires the
entire explicitly configured K×candidate grid, not a hard-coded eight pairs.
Figures and preservation commands work for either dataset. The full raw and
aligned arrays remain distinct. HAM semantics remain diagnoses; JAFFE semantics
are requested expressions. HAM lesion functions are never fed participant IDs.

All new outputs are under `<workspace>/<experiment_id>/runs/K<K>_seed<seed>`.
The workspace must be outside the source checkout, preventing writes into the
historical `research/runs/stability_v1` tree. Never point at a historical attempt
directory as a workspace. A new study cannot silently extend or overwrite the
accepted HAM10000 evidence matrix; report it separately with its own protocol.
