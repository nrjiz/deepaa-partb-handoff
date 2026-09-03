# Journal-extension research area

Everything under `research/` belongs to new journal-extension protocols or future outputs. Historical DATA 2026 preservation remains under `repro/`, and committed historical artifacts remain under `Results_HAM_DAA/`.

Current status: Gate 1 infrastructure is implemented and validated. No primary journal model has been trained and no journal experiment has been executed; smoke artifacts are plumbing checks only.

- `splits/journal_v1/`: accepted immutable lesion-grouped split. Any replacement must be a new `journal_v2`.
- `protocols/` and `configs/`: accepted preregistration and machine-readable record.
- `code/`: manifest adapter, deterministic controls, authoritative-model adapter, alignment, metrics, runner, and output contract.
- `tools/`: single-run, exact-matrix, alignment, held-out evaluation, split-audit, and smoke-test entry points.
- `tests/`: unit, invariant, manifest, parity, and output-safety tests.
- Primary outputs must be unique directories below `research/runs/stability_v1/`; smoke outputs are isolated below `research/smoke_tests/`. Neither may overwrite `Results_HAM_DAA`, `repro/`, or a completed run.
