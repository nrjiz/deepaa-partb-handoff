"""Engineering telemetry only: no model, dataset or random-state access."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd


class Progress:
    def __init__(self):
        self.started = perf_counter()
        self.path = None
        self.pending = []
        self.current_phase = None
        self.phase_started = self.started

    def emit(self, event: str, **fields) -> None:
        record = dict(utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=perf_counter() - self.started,
                      event=event, **fields)
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self.path is None:
            self.pending.append(line)
        else:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

    def bind(self, run_dir: Path) -> None:
        path = run_dir / "progress.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(line + "\n" for line in self.pending))
        self.path = path
        self.pending.clear()

    def phase(self, name: str | None) -> None:
        now = perf_counter()
        if self.current_phase is not None:
            self.emit("phase_end", phase=self.current_phase,
                      duration_seconds=now - self.phase_started)
        self.current_phase = name
        self.phase_started = now
        if name is not None:
            self.emit("phase_start", phase=name)


def write_incremental_history(run_dir: Path, rows: list[dict]) -> None:
    """Atomically replace this active run's history; never a resume checkpoint."""
    temporary = run_dir / "training_history.csv.tmp"
    pd.DataFrame(rows).to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, run_dir / "training_history.csv")
