# Complete dependency chain

All paths below are retained together in this branch. HAM adapters are not
monkeypatched and HAM guards are not weakened.

| Stage | JAFFE entry / retained implementation |
|---|---|
| Configuration, identity, claims, lock, inventory | `jaffe_partb/common.py`, `proposal.json`, `approval.template.json` |
| TIFFs, participant split, expression labels | `jaffe_partb/data.py`, official filename metadata; Pillow, NumPy, TensorFlow |
| Model / objective / original RNG | `research/code/model_adapter.py` → `constants.py` → protected root `daa_ham10000_interpretability_fixed_v2.py` |
| Determinism | `research/code/determinism.py`; Python/NumPy/TF seed, CPU process environment |
| Training / final export | `jaffe_partb/training.py`; shared `_mean_components`, `_evaluate_losses`, `_collect_outputs` in `research/code/runner.py`; original baseline train/eval steps |
| Import closure of shared runner | `research/code/data.py`, `output_contract.py`, `progress.py`, `stability_metrics.py`, `constants.py`, `model_adapter.py`; pandas, h5py, Matplotlib |
| Fresh-process run validation | `jaffe_partb/training.validate_run`; protected model reload and saved tensor hashes, data replay, inventory |
| Validation-only Hungarian | unchanged `research/code/alignment.py`; SciPy linear_sum_assignment |
| Held-out / uncertainty | `jaffe_partb/analysis.py`: explicitly image and participant-cluster functions; no lesion aggregation calls |
| Expression semantics / nearest / decoded comparison | shared `stability_metrics.py` with explicit expression class list; internal nearest `dx` key translated only at call boundary; output key is expression |
| Pair verification / aggregation | `jaffe_partb/analysis.py`; fresh recomputation of fixed validation assignment, all eight pairs required |
| Figures / preservation | `analysis.figures`; Matplotlib. `__main__.preserve`; stdlib ZIP + SHA readback |
| CLI / engineering checks | `jaffe_partb/__main__.py`, `jaffe_partb/tests/test_engineering.py`; pytest |

For historical HAM reproduction, retain `research/tools/run_stability_one.py`,
`align_validation_coefficients.py`, `evaluate_stability_pair.py`,
`research/code/workflow_evidence.py` (validate_run, verify_pair, aggregate, figure),
`recovery.py`, `colab_k7_validation.py`, frozen manifests, config and protocol.
These are all present at the base. They contain HAM-specific identity/lesion/
diagnosis checks and must not be run directly on JAFFE.

Environment: Python3.13.3 and `requirements.txt` (TensorFlow2.21.0, NumPy2.5.2,
pandas3.0.5, Matplotlib3.11.1, h5py3.14.0, Pillow12.3.0, SciPy1.18.1,
pytest9.1.1, Keras3.15.1). Keras is explicitly pinned to the tested runtime.
Fresh package availability and partner OS compatibility are untested.

Additional named HAM experiments use the original `load_journal_v1` and
`make_tf_dataset` through `data.audit_ham/load_ham`, and the unchanged lesion
functions through `analysis.compute_pair`. Additional K/seeds/reference selection
are configuration inputs only; no inherited constants or model code are edited.
See NEW_EXPERIMENTS.md for separate K3 examples and evaluation-unit dispatch.

External items intentionally absent: JAFFE images/terms acceptance, actual image
manifest, scientific approval, new results/weights, independent backup service.
HAM run receipts and completed evidence live outside this checkout; see provenance.
They are not silently substituted with the older tracked conference artifacts.
