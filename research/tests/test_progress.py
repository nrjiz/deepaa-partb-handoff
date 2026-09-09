import json

import pandas as pd
import pytest

from research.code.progress import Progress, write_incremental_history
from research.code.runner import _file_sha256, _write_artifact_inventory


def test_progress_phases_flush_and_inventory(tmp_path, capsys):
    progress = Progress()
    progress.phase("preflight")
    assert '"phase": "preflight"' in capsys.readouterr().out
    progress.bind(tmp_path)
    for phase in ["loading", "training", "evaluation", "export"]:
        progress.phase(phase)
    progress.phase(None)
    rows = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "phase_end"]) == 5
    assert all(r.get("duration_seconds", 0) >= 0 for r in rows)
    _write_artifact_inventory(tmp_path)
    inventory = json.loads((tmp_path / "artifact_inventory.json").read_text())
    assert inventory["artifacts"][0]["sha256"] == _file_sha256(tmp_path / "progress.jsonl")
    with pytest.raises(FileExistsError):
        Progress().bind(tmp_path)


def test_history_incremental_same_schema(tmp_path):
    rows = [{"epoch": 1, "train_total": 1.25}]
    write_incremental_history(tmp_path, rows)
    assert pd.read_csv(tmp_path / "training_history.csv").epoch.tolist() == [1]
    rows.append({"epoch": 2, "train_total": 1.0})
    write_incremental_history(tmp_path, rows)
    assert pd.read_csv(tmp_path / "training_history.csv").epoch.tolist() == [1, 2]
    assert not (tmp_path / "training_history.csv.tmp").exists()
