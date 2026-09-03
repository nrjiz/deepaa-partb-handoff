from __future__ import annotations

from research.code.determinism import DeterminismRecord, process_environment


def test_fresh_process_environment_and_rng_record_cover_all_streams() -> None:
    assert process_environment(211) == {
        "PYTHONHASHSEED": "211",
        "TF_DETERMINISTIC_OPS": "1",
    }
    record = DeterminismRecord(
        seed=211,
        python_hash_seed="211",
        python_random_seed=211,
        numpy_seed=211,
        tensorflow_seed=211,
        keras_seed=211,
        dataset_shuffle_seed=211,
        model_initialization_seed=211,
        posterior_sampling_seed=211,
        deterministic_ops_requested=True,
        deterministic_ops_enabled=True,
        warnings=[],
    ).to_dict()
    for field in (
        "python_random_seed",
        "numpy_seed",
        "tensorflow_seed",
        "keras_seed",
        "dataset_shuffle_seed",
        "model_initialization_seed",
        "posterior_sampling_seed",
    ):
        assert record[field] == 211
