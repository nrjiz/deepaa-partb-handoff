from __future__ import annotations

from pathlib import Path

import pytest

from research.code.output_contract import (
    OutputContractError,
    mark_completed,
    require_scoped_output,
    reserve_run_directory,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def test_completed_output_refuses_overwrite_and_override_uses_new_sibling(tmp_path: Path) -> None:
    requested = tmp_path / "seed_113"
    reserved = reserve_run_directory(requested)
    mark_completed(reserved, {"status": "COMPLETED"})
    with pytest.raises(OutputContractError, match="Refusing to overwrite completed"):
        reserve_run_directory(requested)
    rerun = reserve_run_directory(requested, allow_completed_rerun=True)
    assert rerun != requested
    assert rerun.name.startswith("seed_113_rerun_")
    assert (rerun / ".RUNNING").exists()


def test_incomplete_output_refuses_overwrite(tmp_path: Path) -> None:
    requested = tmp_path / "incomplete"
    reserve_run_directory(requested)
    with pytest.raises(OutputContractError, match="non-empty/incomplete"):
        reserve_run_directory(requested)


def test_primary_and_smoke_output_scopes() -> None:
    primary = REPOSITORY / "research" / "runs" / "stability_v1" / "K5" / "seed_113"
    smoke = REPOSITORY / "research" / "smoke_tests" / "check"
    assert require_scoped_output(REPOSITORY, primary, smoke=False) == primary.resolve()
    assert require_scoped_output(REPOSITORY, smoke, smoke=True) == smoke.resolve()
    with pytest.raises(OutputContractError):
        require_scoped_output(REPOSITORY, REPOSITORY / "Results_HAM_DAA" / "bad", smoke=False)
    with pytest.raises(OutputContractError, match="unique child"):
        require_scoped_output(
            REPOSITORY, REPOSITORY / "research" / "runs" / "stability_v1", smoke=False
        )
