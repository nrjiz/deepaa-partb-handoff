# Baseline invariant verification

Result: **30/30 passed**, 0 failed. No training was performed.

| Check | Status | Details |
|---|---|---|
| authoritative_source_hash | PASS | expected=6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d; actual=6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d |
| test_manifest_count | PASS | expected=1001; actual=1001 |
| test_manifest_hash | PASS | expected=e24a78ffb22be7f0dfe8ad63b4c0d87f7ac3eaf890658c474a9c55a4cc1c4b63; actual=e24a78ffb22be7f0dfe8ad63b4c0d87f7ac3eaf890658c474a9c55a4cc1c4b63 |
| train_manifest_count | PASS | expected=8013; actual=8013 |
| train_manifest_hash | PASS | expected=5573c4370a1133a7e8272d708f8c9f90b595309601a496c4585471d0d6d47e44; actual=5573c4370a1133a7e8272d708f8c9f90b595309601a496c4585471d0d6d47e44 |
| validation_manifest_count | PASS | expected=1001; actual=1001 |
| validation_manifest_hash | PASS | expected=e989f41b2d96be20b1ba101f8cf8a17004df02e224510b1b5e1e1d5459185272; actual=e989f41b2d96be20b1ba101f8cf8a17004df02e224510b1b5e1e1d5459185272 |
| image_id_disjoint_train_validation | PASS | overlap_count=0 |
| image_id_disjoint_train_test | PASS | overlap_count=0 |
| image_id_disjoint_validation_test | PASS | overlap_count=0 |
| K3_checkpoint_load | PASS | Results_HAM_DAA/20260508_002312/daa_ham_weights.weights.h5 |
| K3_simplex_shape | PASS | actual=[3, 2]; expected=[3, 2] |
| K3_coefficients_nonnegative | PASS | minimum=1.09327318e-10 |
| K3_coefficient_rows_sum_to_one | PASS | maximum_absolute_error=1.1920929e-07; tolerance=1e-6 |
| K3_mu_equals_a_times_z_fixed | PASS | shape=[2, 2]; maximum_absolute_error=0; tolerance=1e-6 |
| K5_checkpoint_load | PASS | Results_HAM_DAA/20260507_131506/daa_ham_weights.weights.h5 |
| K5_simplex_shape | PASS | actual=[5, 4]; expected=[5, 4] |
| K5_coefficients_nonnegative | PASS | minimum=1.97709369e-06 |
| K5_coefficient_rows_sum_to_one | PASS | maximum_absolute_error=1.1920929e-07; tolerance=1e-6 |
| K5_mu_equals_a_times_z_fixed | PASS | shape=[2, 4]; maximum_absolute_error=2.98023224e-08; tolerance=1e-6 |
| K7_checkpoint_load | PASS | Results_HAM_DAA/20260508_104147/daa_ham_weights.weights.h5 |
| K7_simplex_shape | PASS | actual=[7, 6]; expected=[7, 6] |
| K7_coefficients_nonnegative | PASS | minimum=2.53141792e-07 |
| K7_coefficient_rows_sum_to_one | PASS | maximum_absolute_error=0; tolerance=1e-6 |
| K7_mu_equals_a_times_z_fixed | PASS | shape=[2, 6]; maximum_absolute_error=5.96046448e-08; tolerance=1e-6 |
| K9_checkpoint_load | PASS | Results_HAM_DAA/20260509_112437/daa_ham_weights.weights.h5 |
| K9_simplex_shape | PASS | actual=[9, 8]; expected=[9, 8] |
| K9_coefficients_nonnegative | PASS | minimum=6.14347115e-08 |
| K9_coefficient_rows_sum_to_one | PASS | maximum_absolute_error=1.1920929e-07; tolerance=1e-6 |
| K9_mu_equals_a_times_z_fixed | PASS | shape=[2, 8]; maximum_absolute_error=1.49011612e-08; tolerance=1e-6 |
