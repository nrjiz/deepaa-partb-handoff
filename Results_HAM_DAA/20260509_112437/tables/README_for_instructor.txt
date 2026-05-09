These outputs answer the requirement: DeepAA interpretation for EVERY HAM10000 skin lesion type.

Use table2_skin_lesion_archetype_profiles.csv to show the average archetype contribution for each lesion type.
Use table3_class_distribution_per_archetype_percent.csv to show which lesion types each archetype mainly represents.
Use per_image_archetype_interpretation.csv as per-image proof: image id, true label, predicted label, reconstruction error, and all A weights.
Use interpretability_examples/*.png as visual proof: top real images per lesion type and archetype.
For K comparison, run four experiments: --latent_dim 2, --latent_dim 4, --latent_dim 6, --latent_dim 8. These correspond to K=3,5,7,9.
