"""Non-overwriting run-directory and completion-marker contract."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class OutputContractError(RuntimeError):
    """Raised when a run could overwrite or mix with prior output."""


def require_scoped_output(repository: Path, requested: Path, *, smoke: bool) -> Path:
    repository = repository.resolve()
    requested = requested.expanduser().resolve()
    allowed = (
        repository / "research" / "smoke_tests"
        if smoke
        else repository / "research" / "runs" / "stability_v1"
    )
    allowed = allowed.resolve()
    if not requested.is_relative_to(allowed):
        raise OutputContractError(f"Output must be below {allowed}; got {requested}")
    if requested == allowed:
        raise OutputContractError(f"Output must be a unique child directory below {allowed}")
    return requested


def reserve_run_directory(
    requested: Path,
    *,
    allow_completed_rerun: bool = False,
) -> Path:
    """Reserve a new run directory; explicit reruns receive a new sibling path."""
    requested = requested.resolve()
    if requested.exists() and any(requested.iterdir()):
        completed = requested / "COMPLETED.json"
        if completed.exists() and allow_completed_rerun:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            requested = requested.with_name(f"{requested.name}_rerun_{stamp}")
        else:
            state = "completed" if completed.exists() else "non-empty/incomplete"
            raise OutputContractError(f"Refusing to overwrite {state} run directory: {requested}")
    requested.mkdir(parents=True, exist_ok=False) if not requested.exists() else None
    marker = requested / ".RUNNING"
    try:
        with marker.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(datetime.now(timezone.utc).isoformat() + "\n")
    except FileExistsError as exc:
        raise OutputContractError(f"Run directory is already reserved: {requested}") from exc
    return requested


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def mark_completed(run_dir: Path, payload: dict) -> None:
    if (run_dir / "COMPLETED.json").exists():
        raise OutputContractError(f"Completion marker already exists: {run_dir}")
    write_json(run_dir / "COMPLETED.json", payload)
    running = run_dir / ".RUNNING"
    if running.exists():
        running.unlink()


def mark_failed(run_dir: Path, payload: dict) -> None:
    write_json(run_dir / "FAILED.json", payload)
    running = run_dir / ".RUNNING"
    if running.exists():
        running.unlink()
