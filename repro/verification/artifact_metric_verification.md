# Artifact-only metric verification

Metrics were recomputed only from committed CSV artifacts; historical result files were not changed. The tolerance for numeric CSV recomputation is `1e-7`. Published expected values are rounded and are shown for provenance, not judged with that tolerance.

| K | Metric | Published (rounded) | Committed | Recomputed | Absolute difference | Status |
|---:|---|---:|---:|---:|---:|---|
| 3 | mse | 0.008679 | 0.00867916222187 | 0.00867916222187 | 3.64291929955e-17 | MATCH |
| 3 | accuracy | 0.666334 | 0.666333666334 | 0.666333666334 | 0 | MATCH |
| 3 | coefficient_variance | 0.022459 | 0.0224593565305 | 0.0224593565305 | 5.89805981832e-17 | MATCH |
| 3 | normalized_entropy | 0.897674 | 0.897673815136 | 0.897673796253 | 1.88835377335e-08 | MATCH |
| 3 | sparsity | 0.102326 | 0.102326203747 | 0.102326203747 | 1.52655665886e-16 | MATCH |
| 3 | average_active_archetypes | N/A | 2.82517482517 | 2.82517482517 | 0 | MATCH |
| 3 | kl | 9.05205 | 9.05204951763 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 3 | simplex_penalty | 0.33327 | 0.333269857569 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 5 | mse | 0.006598 | 0.00659770263195 | 0.00659770263195 | 3.29597460436e-17 | MATCH |
| 5 | accuracy | 0.672328 | 0.672327672328 | 0.672327672328 | 0 | MATCH |
| 5 | coefficient_variance | 0.010752 | 0.0107522122863 | 0.0107522122863 | 9.02056207508e-17 | MATCH |
| 5 | normalized_entropy | 0.913765 | 0.913765348514 | 0.913765343431 | 5.08313136027e-09 | MATCH |
| 5 | sparsity | 0.086235 | 0.086234656569 | 0.086234656569 | 2.35922392733e-16 | MATCH |
| 5 | average_active_archetypes | N/A | 4.19380619381 | 4.19380619381 | 0 | MATCH |
| 5 | kl | 18.027552 | 18.0275518298 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 5 | simplex_penalty | 0.558307 | 0.558307135478 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 7 | mse | 0.004913 | 0.00491323134252 | 0.00491323134252 | 2.08166817117e-17 | MATCH |
| 7 | accuracy | 0.695305 | 0.695304695305 | 0.695304695305 | 0 | MATCH |
| 7 | coefficient_variance | 0.006778 | 0.00677815151413 | 0.00677815151413 | 2.25514051877e-17 | MATCH |
| 7 | normalized_entropy | 0.912072 | 0.912071763051 | 0.912071760847 | 2.2034811753e-09 | MATCH |
| 7 | sparsity | 0.087928 | 0.0879282391525 | 0.0879282391526 | 3.1918911958e-16 | MATCH |
| 7 | average_active_archetypes | N/A | 4.57842157842 | 4.57842157842 | 0 | MATCH |
| 7 | kl | 27.029241 | 27.0292413831 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 7 | simplex_penalty | 0.653163 | 0.653162874281 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 9 | mse | 0.004175 | 0.00417452943508 | 0.00417452943508 | 3.29597460436e-17 | MATCH |
| 9 | accuracy | 0.709291 | 0.709290709291 | 0.709290709291 | 0 | MATCH |
| 9 | coefficient_variance | 0.00435 | 0.00434975303847 | 0.00434975303847 | 7.1991024253e-17 | MATCH |
| 9 | normalized_entropy | 0.916249 | 0.916248717534 | 0.916248714827 | 2.70719124895e-09 | MATCH |
| 9 | sparsity | 0.083751 | 0.0837512851734 | 0.0837512851734 | 3.05311331772e-16 | MATCH |
| 9 | average_active_archetypes | N/A | 4.37362637363 | 4.37362637363 | 0 | MATCH |
| 9 | kl | 36.024504 | 36.0245043039 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |
| 9 | simplex_penalty | 0.710956 | 0.710955869406 | N/A | N/A | NOT_RECOMPUTABLE_FROM_ARTIFACTS |

Archetype usage was checked against the historical one-decimal strings with a 0.05 percentage-point rounding allowance. Every cell of each class-by-archetype distribution table was also recomputed; those detailed comparisons are in the CSV.

KL cannot be independently recovered from CSVs because per-image posterior `logvar` was not saved. The simplex/archetype penalty cannot be recovered because batch-derived `z_pred` was not saved. Both require checkpoint evaluation.

Status counts: MATCH=192, MATCH_ROUNDED=24, NOT_RECOMPUTABLE_FROM_ARTIFACTS=8.
