"""Bounded engineering comparison of old/new loop ASTs; never a primary run.

Two epochs of 65 training and 10 evaluation images, all from frozen TRAIN.
Runs fresh workers sequentially. No execute_run guard is patched or invoked.
The loop statements are extracted, not reimplemented. Outputs must be outside
research/runs; existing evidence directories are refused.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD = "42ec30f84cf826a2a3465fbb0778af4b04fc0bd5"


def worker(mode, out):
    import numpy as np
    import pandas as pd
    from research.code.data import load_journal_v1
    from research.code.determinism import configure_determinism
    from research.code.model_adapter import build_model
    from research.code.progress import Progress
    from research.code import runner

    source = (subprocess.check_output(["git", "show", f"{OLD}:research/code/runner.py"], cwd=ROOT).decode()
              if mode == "old" else (ROOT / "research/code/runner.py").read_text())
    module = types.ModuleType(f"research.code.equivalence_{mode}")
    sys.modules[module.__name__] = module
    exec(compile(source, f"<{mode}_runner>", "exec"), module.__dict__)
    split = ROOT / "research/splits/journal_v1"
    dataset = Path("C:/Users/UserPc/Downloads/archive")
    frames = load_journal_v1(split / "train.csv", split / "validation.csv", split / "test.csv",
                            [dataset / "HAM10000_images_part_1", dataset / "HAM10000_images_part_2"])
    configure_determinism(211)
    import tensorflow as tf
    baseline, model, cfg = build_model(ROOT, 5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
    trace = []
    batch_hashes = []
    history_at_next_epoch = []
    train_step = baseline.train_step

    def observed_step(model, optimizer, images, labels, cfg):
        if int(optimizer.iterations.numpy()) == 3 and mode == "new":
            history = pd.read_csv(out / "training_history.csv")
            assert history.epoch.tolist() == [1]
            history_at_next_epoch.append(True)
        batch_hashes.append(hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest())
        values = train_step(model, optimizer, images, labels, cfg)
        trace.append([float(x.numpy()) for x in values])
        return values

    baseline.train_step = observed_step
    body = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "execute_run").body
    body = next(n for n in body if isinstance(n, ast.Try)).body
    def assigned(n, name):
        return isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    begin = next(i for i, n in enumerate(body) if assigned(n, "train_dataset"))
    end = next(i for i, n in enumerate(body) if assigned(n, "checkpoint"))
    loop = ast.Module(body=body[begin:end], type_ignores=[])
    progress = Progress()
    progress.bind(out)
    env = dict(module.__dict__, active_frames={"train": frames["train"].iloc[:65].copy(),
               "validation": frames["train"].iloc[65:75].copy()},
               spec=types.SimpleNamespace(seed=211, smoke_test=True, smoke_epochs=2),
               baseline=baseline, model=model, cfg=cfg, optimizer=optimizer,
               run_dir=out, progress=progress)
    exec(compile(loop, f"<{mode}_training_loop>", "exec"), env)
    progress.phase(None)
    runner._write_artifact_inventory(out)
    if mode == "new":
        assert history_at_next_epoch == [True]
        records = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
        assert [r["epoch"] for r in records if r["event"] == "epoch_end"] == [1, 2]
        assert [r["batch"] for r in records if r["event"] == "batch"] == [1, 3, 1, 3]
        assert all(r["remaining_training_loop_seconds"] >= 0 for r in records if r["event"] == "epoch_end")
    arrays = {f"weight_{i}": v.numpy() for i, v in enumerate(model.weights)}
    arrays.update({f"optimizer_{i}": v.numpy() for i, v in enumerate(optimizer.variables)})
    arrays["losses"] = np.array(trace)
    np.savez(out / "state.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(dict(scientific_evidence=False, mode=mode,
        optimizer_iterations=int(optimizer.iterations.numpy()), batch_hashes=batch_hashes,
        incremental_history_checked=bool(history_at_next_epoch), tensorflow=tf.__version__,
        keras=tf.keras.__version__, python=sys.version, devices=[str(d) for d in tf.config.list_physical_devices()]), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=["old", "new"])
    args = parser.parse_args()
    out = args.output.resolve()
    if out.is_relative_to(ROOT / "research/runs"):
        raise ValueError("Engineering evidence cannot be stored under primary runs")
    out.mkdir(parents=True, exist_ok=False)
    if args.worker:
        worker(args.worker, out)
        return
    environment = dict(os.environ, PYTHONHASHSEED="211", TF_DETERMINISTIC_OPS="1", PYTHONDONTWRITEBYTECODE="1")
    for mode in ["old", "new"]:
        with (out / f"{mode}.log").open("x", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-B", __file__, "--worker", mode, "--output", str(out / mode)],
                           cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    import numpy as np
    import pandas as pd
    old, new = (np.load(out / mode / "state.npz") for mode in ["old", "new"])
    assert set(old.files) == set(new.files)
    differences = {}
    for key in old.files:
        assert np.isfinite(old[key]).all() and np.isfinite(new[key]).all()
        # Stronger than existing numerical tolerances: exact same-host equality.
        np.testing.assert_array_equal(old[key], new[key], err_msg=key)
        differences[key] = float(np.max(np.abs(old[key] - new[key])))
    pd.testing.assert_frame_equal(*(pd.read_csv(out / mode / "training_history.csv") for mode in ["old", "new"]))
    summaries = [json.loads((out / mode / "summary.json").read_text()) for mode in ["old", "new"]]
    assert summaries[0]["batch_hashes"] == summaries[1]["batch_hashes"]
    assert all(s["optimizer_iterations"] == 6 for s in summaries)
    result = dict(status="PASS", scientific_evidence=False, baseline_revision=OLD,
                  epochs=2, training_rows=65, evaluation_rows_from_train=10,
                  optimizer_steps_per_path=6, exact_batch_order=True, exact_history=True,
                  maximum_absolute_differences=differences,
                  limitation="Bounded same-host loop comparison, not a full primary trajectory or performance benchmark")
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
