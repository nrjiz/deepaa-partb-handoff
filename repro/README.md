# DATA 2026 baseline reproduction

This directory preserves and evaluates the high-confidence working conference baseline at commit `29acec505161530f9ea811534b05dd52d8df9899`. The assignment is evaluation-only: none of these utilities trains a model or changes the authoritative source or historical `Results_HAM_DAA` artifacts.

The baseline is high-confidence because Git history and all published rounded K=3/5/7/9 values align with the committed artifacts, and all four checkpoints reproduce the committed metrics. It is not formally certain because there is no conference tag/release, submitted archive, exact run transcript, or author confirmation.

## Contents

- `baseline_inventory.md`: baseline identity, hashes, result/checkpoint inventory, and drift boundaries.
- `baseline_artifact_hashes.csv`: SHA-256 and byte size for all 226 committed May result files.
- `configs/conference_runs.json`: evidence-classified configuration reconstruction.
- `reproduction_environment.md` and `requirements_reproduction.txt`: the environment used now, not an asserted historical lock.
- `splits/conference_image_level/`: immutable reconstruction of the historical 8,013/1,001/1,001 image split and lesion-overlap audit.
- `verification/`: artifact-only, checkpoint, and invariant results.
- `tools/`: small reproduction utilities that import or mirror the baseline only where needed.

## Re-run commands

Activate or call the isolated Python executable at `C:\Users\UserPc\Desktop\ai\.venvs\deepaa_data2026_repro\Scripts\python.exe`, then run from the repository root:

```powershell
python repro\tools\inventory_baseline.py --repository . --output repro\baseline_artifact_hashes.csv
python repro\tools\reconstruct_conference_split.py --metadata 'C:\Users\UserPc\Downloads\archive\HAM10000_metadata.csv' --image-dirs 'C:\Users\UserPc\Downloads\archive\HAM10000_images_part_1' 'C:\Users\UserPc\Downloads\archive\HAM10000_images_part_2' --output-dir repro\splits\conference_image_level
python repro\tools\verify_artifact_metrics.py --results-root Results_HAM_DAA --output-dir repro\verification
python repro\tools\evaluate_checkpoints.py --repository . --test-manifest repro\splits\conference_image_level\test.csv --image-dirs 'C:\Users\UserPc\Downloads\archive\HAM10000_images_part_1' 'C:\Users\UserPc\Downloads\archive\HAM10000_images_part_2' --output-dir repro\verification
python repro\tools\verify_baseline_invariants.py --repository . --split-dir repro\splits\conference_image_level --output-dir repro\verification
```

The dataset is referenced in its existing local location. It is not copied into the repository.

## Historical entropy behavior

The final May baseline does **not** train with coefficient-entropy stabilization or coefficient temperature scaling. Entropy and `1 - entropy` sparsity are post-training evaluation metrics. Earlier January history contained entropy/temperature training logic; this reproduction does not restore it.

## Related implementation

`C:\Users\UserPc\Documents\github\ham10000_aanet` is a **related / non-authoritative historical implementation** in deterministic PyTorch. It is not used by this reproduction.
