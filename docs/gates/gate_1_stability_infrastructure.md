# Gate 1 — Stability infrastructure

Status: **PREPARED; NOT APPROVED FOR TRAINING**.

Gate 0 passed with documented limitations. Gate 1 preparation now includes:

- `PROPOSED JOURNAL SPLIT V1`: 8,011 train, 1,002 validation, and 1,002 test images; 5,975/747/748 lesions; zero lesion overlap; all 17 split invariants passed.
- Pre-registered primary matrix: K=`{5,7}`, seeds=`{113,211,307,401,503}`, ten planned runs.
- Predetermined seed-113 reference within each K.
- Validation-only activation-signature Hungarian matching and test-only stability evaluation.
- Frozen baseline architecture/objective and explicit planned training configuration.
- Randomness-control requirements and raw/aligned artifact preservation rules.
- Local compute audit: TensorFlow sees no GPU and only about 12.75 GiB disk was free, so ten local CPU runs are not presently practical.

## Exit criteria

- Approved seed list and multi-run launcher/configuration without overwriting outputs.
- Coefficients and semantic artifacts extracted on fixed validation/reference and test sets.
- Hungarian matching implemented with approved activation-signature costs and edge-case handling.
- Synthetic permutation tests recover known mappings.
- Alignment learned without test-set use.
- Held-out stability metrics, uncertainty summaries, and audit artifacts produced.
- Run registry complete for every included/excluded seed.

## Remaining blockers before execution

- Senior-developer review/acceptance of journal_v1 statistics and manifest hashes.
- Approved definition and fallback for constant or nearly inactive activation columns.
- Approved decoded-archetype visual similarity metric.
- Manifest-aware training launcher/data adapter, without modifying the historical source, plus overwrite protection and synthetic alignment tests.
- A practical training target: substantially more local free disk plus acceptable CPU runtime, or an approved external GPU environment.

No stability result may compare raw archetype IDs across independent runs as if IDs were identifiable.
