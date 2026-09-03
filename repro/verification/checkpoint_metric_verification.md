# Checkpoint evaluation verification

All four committed HDF5 checkpoints were loaded into the unmodified baseline model and evaluated with deterministic inference (`training=False`) on the reconstructed 1,001-image test split. No optimization or training was performed.

Numeric match tolerance: `1e-05` absolute. Accuracy is also reported as a fraction of 1,001 images.

| K | Metric | Published (rounded) | Committed | Reproduced | Absolute difference | Status |
|---:|---|---:|---:|---:|---:|---|
| 3 | mse | 0.008679 | 0.00867916222187 | 0.00867916271091 | 4.89e-10 | MATCH |
| 3 | kl | 9.05205 | 9.05204951763 | 9.05204951763 | 0 | MATCH |
| 3 | simplex_penalty | 0.33327 | 0.333269857569 | 0.333269857569 | 5.55e-17 | MATCH |
| 3 | accuracy | 0.666334 | 0.666333666334 | 0.666333666334 | 0 | MATCH |
| 3 | coefficient_variance | 0.022459 | 0.0224593565305 | 0.0224593412131 | 1.53e-08 | MATCH |
| 3 | normalized_entropy | 0.897674 | 0.897673815136 | 0.897673815136 | 0 | MATCH |
| 3 | sparsity | 0.102326 | 0.102326203747 | 0.102326184864 | 1.89e-08 | MATCH |
| 3 | average_active_archetypes | 2.82517482517 | 2.82517482517 | 2.82517482517 | 0 | MATCH |
| 5 | mse | 0.006598 | 0.00659770263195 | 0.00659770285711 | 2.25e-10 | MATCH |
| 5 | kl | 18.027552 | 18.0275518298 | 18.0275518298 | 0 | MATCH |
| 5 | simplex_penalty | 0.558307 | 0.558307135478 | 0.558307135478 | 0 | MATCH |
| 5 | accuracy | 0.672328 | 0.672327672328 | 0.672327672328 | 0 | MATCH |
| 5 | coefficient_variance | 0.010752 | 0.0107522122863 | 0.0107522131875 | 9.01e-10 | MATCH |
| 5 | normalized_entropy | 0.913765 | 0.913765348514 | 0.913765348514 | 1.11e-16 | MATCH |
| 5 | sparsity | 0.086235 | 0.086234656569 | 0.0862346514858 | 5.08e-09 | MATCH |
| 5 | average_active_archetypes | 4.19380619381 | 4.19380619381 | 4.19380619381 | 0 | MATCH |
| 7 | mse | 0.004913 | 0.00491323134252 | 0.00491323089227 | 4.5e-10 | MATCH |
| 7 | kl | 27.029241 | 27.0292413831 | 27.0292413831 | 3.55e-15 | MATCH |
| 7 | simplex_penalty | 0.653163 | 0.653162874281 | 0.653162874281 | 0 | MATCH |
| 7 | accuracy | 0.695305 | 0.695304695305 | 0.695304695305 | 0 | MATCH |
| 7 | coefficient_variance | 0.006778 | 0.00677815151413 | 0.00677815172821 | 2.14e-10 | MATCH |
| 7 | normalized_entropy | 0.912072 | 0.912071763051 | 0.912071763051 | 0 | MATCH |
| 7 | sparsity | 0.087928 | 0.0879282391525 | 0.0879282369491 | 2.2e-09 | MATCH |
| 7 | average_active_archetypes | 4.57842157842 | 4.57842157842 | 4.57842157842 | 0 | MATCH |
| 9 | mse | 0.004175 | 0.00417452943508 | 0.00417452910915 | 3.26e-10 | MATCH |
| 9 | kl | 36.024504 | 36.0245043039 | 36.0245043039 | 0 | MATCH |
| 9 | simplex_penalty | 0.710956 | 0.710955869406 | 0.710955869406 | 0 | MATCH |
| 9 | accuracy | 0.709291 | 0.709290709291 | 0.709290709291 | 0 | MATCH |
| 9 | coefficient_variance | 0.00435 | 0.00434975303847 | 0.00434975139797 | 1.64e-09 | MATCH |
| 9 | normalized_entropy | 0.916249 | 0.916248717534 | 0.916248717534 | 0 | MATCH |
| 9 | sparsity | 0.083751 | 0.0837512851734 | 0.0837512824662 | 2.71e-09 | MATCH |
| 9 | average_active_archetypes | 4.37362637363 | 4.37362637363 | 4.37362637363 | 0 | MATCH |

## Evaluation details

- K=3: checkpoint `Results_HAM_DAA/20260508_002312/daa_ham_weights.weights.h5`, 32 batches at batch size 32, simplex shape [3, 2], minimum coefficient 1.03e-23, maximum row-sum error 1.19e-07.
- K=5: checkpoint `Results_HAM_DAA/20260507_131506/daa_ham_weights.weights.h5`, 32 batches at batch size 32, simplex shape [5, 4], minimum coefficient 0.000221, maximum row-sum error 1.79e-07.
- K=7: checkpoint `Results_HAM_DAA/20260508_104147/daa_ham_weights.weights.h5`, 32 batches at batch size 32, simplex shape [7, 6], minimum coefficient 3.06e-05, maximum row-sum error 2.38e-07.
- K=9: checkpoint `Results_HAM_DAA/20260509_112437/daa_ham_weights.weights.h5`, 32 batches at batch size 32, simplex shape [9, 8], minimum coefficient 2.86e-05, maximum row-sum error 2.38e-07.

The KL and simplex values intentionally reproduce the historical implementation's mean of batch means. The simplex term is batch-composition-dependent because `z_pred` uses a softmax across each batch; changing test order or batch size changes that metric.
