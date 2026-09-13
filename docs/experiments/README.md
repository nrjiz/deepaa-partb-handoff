# Experiment documentation

These files include accepted protocols, future proposals and dated execution records. As of 2026-09-13 the experiment registry contains two completed K5 runs and one validation-aligned held-out stability comparison; other proposed experiments are not thereby approved or executed.

Every run must receive an experiment-registry entry before its outputs are used. Record the split manifest ID, seed, manually entered Git commit hash, full configuration, checkpoint, output paths, status, and failures. Never overwrite a prior run directory. Derived tables and figures must trace back to immutable run IDs.

Shared rules:

- Use one frozen train/validation/test split family across matched comparisons.
- Prefer lesion-grouped, diagnosis-stratified splitting for HAM10000 and record the algorithm/version.
- Fit models on training data; use validation/reference data for alignment and tuning; use test data once for final evaluation.
- Report all planned seeds and uncertainty, not only the best run.
- Preserve raw outputs and generate aggregates separately.
- Do not use test diagnosis labels to optimize local inference or counterfactual proximity.
- Mark exploratory analyses as exploratory.
