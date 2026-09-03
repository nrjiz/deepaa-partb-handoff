# journal_v1 leakage and invariant audit

Status: **PROPOSED JOURNAL SPLIT V1 — pending senior-developer review**.

The split was generated once using the predetermined rule. No alternative seeds or fold assignments were searched for class balance.

## Partition counts

| Partition | Images | Lesions | Approximate image proportion |
|---|---:|---:|---:|
| train | 8011 | 5975 | 79.9900% |
| validation | 1002 | 747 | 10.0050% |
| test | 1002 | 748 | 10.0050% |

## Invariants

| Check | Status | Details |
|---|---|---|
| metadata_row_count | PASS | actual=10015 |
| metadata_image_id_unique | PASS | unique=10015 |
| every_image_exactly_once | PASS | combined_rows=10015; unique=10015 |
| no_images_lost | PASS | missing=0 |
| no_unknown_images_in_manifests | PASS | unknown=0 |
| all_metadata_images_available | PASS | missing_files=0 |
| no_unknown_available_image_files | PASS | extra_files=0 |
| no_duplicate_image_files_across_sources | PASS | duplicate_file_ids=0 |
| image_ids_disjoint_train_validation | PASS | overlap=0 |
| lesion_ids_disjoint_train_validation | PASS | overlap=0 |
| image_ids_disjoint_train_test | PASS | overlap=0 |
| lesion_ids_disjoint_train_test | PASS | overlap=0 |
| image_ids_disjoint_validation_test | PASS | overlap=0 |
| lesion_ids_disjoint_validation_test | PASS | overlap=0 |
| all_classes_represented_train | PASS | missing=[] |
| all_classes_represented_validation | PASS | missing=[] |
| all_classes_represented_test | PASS | missing=[] |

All three pairwise lesion intersections are empty; therefore measured lesion-level leakage is zero. Every diagnosis class occurs in every partition.
