# Historical lesion-level overlap audit

This measures leakage caused by the conference baseline's diagnosis-stratified, image-level split. It does not alter the split and does not by itself invalidate the historical results.

## Pairwise overlap

| Partitions | Shared lesion IDs | Left affected images (%) | Right affected images (%) |
|---|---:|---:|---:|
| train / validation | 357 | 482 (6.0152%) | 372 (37.1628%) |
| train / test | 362 | 479 (5.9778%) | 376 (37.5624%) |
| validation / test | 50 | 51 (5.0949%) | 53 (5.2947%) |

Lesions appearing in all three partitions: **22**.
Unique lesion IDs crossing at least one boundary: **725**.
Images belonging to those lesions: **1737**.

## Any-boundary affected images

| Partition | Affected images | Percent of partition |
|---|---:|---:|
| train | 932 | 11.6311% |
| validation | 400 | 39.9600% |
| test | 405 | 40.4595% |

## Overlapping lesions by diagnosis

| Diagnosis | Unique overlapping lesion IDs |
|---|---:|
| akiec | 29 |
| bcc | 53 |
| bkl | 109 |
| df | 10 |
| mel | 145 |
| nv | 366 |
| vasc | 13 |

Machine-readable detail: `leakage_details.csv` (SHA-256 `d801219bbb2da99ecd60afbb88b2f0bb3cefce501181d02b040d5a872e77b905`).
