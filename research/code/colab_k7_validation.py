"""Schema-aware CSV decoding only for ordinary K7/latent6/seed113 validation."""
from contextlib import contextmanager
from pathlib import Path
import csv

from .workflow_evidence import read, require


@contextmanager
def semantic_reason_loader(path):
    """Decode only empty reason cells at this exact path, in a dedicated worker.

    Numeric parsing and comparison tolerances are unchanged. Restore the original
    reader on every exit. Never use this adapter around a recovery validation.
    """
    import pandas as pd
    target = Path(path).resolve()
    original = pd.read_csv

    def read_csv(source, *args, **kwargs):
        if isinstance(source, (str, Path)) and Path(source).resolve() == target:
            require("converters" not in kwargs, "Unexpected existing CSV converters")
            kwargs["converters"] = {"reason": lambda value: None if value == "" else value}
        return original(source, *args, **kwargs)

    pd.read_csv = read_csv
    try:
        yield
    finally:
        pd.read_csv = original


def validate_k7_reference(root, run, approved_heads):
    """Call the unchanged hash-bound validator after strict identity scoping."""
    from .workflow_evidence import validate_run
    run = Path(run)
    require(not (run / "RECOVERED_EXPORT.json").exists(), "K7 adapter refuses recovered exports")
    config = read(run / "configuration.json")
    require((config.get("K"), config.get("latent_dimension"), config.get("seed")) == (7, 6, 113),
            "K7 adapter permits only K7/latent6/seed113")
    table_path = run / "test_archetype_class_distribution_raw.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle), [])
    required = {"raw_archetype", "dx", "coefficient_weighted_probability", "test_dx_prevalence",
                "enrichment_ratio", "coefficient_mass", "status", "reason"}
    require(required.issubset(columns), "Missing required K7 semantic fields")
    with semantic_reason_loader(table_path):
        return validate_run(root, run, 7, 113, approved_heads)
