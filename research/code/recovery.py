"""D-019 seed401 export recovery. No training, optimizer or original writes.

The sole partial-run exception is the specifically authorized failed attempt.
The receipt is separate from COMPLETED, bound to originals, code and registration.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .workflow_evidence import require, read, write_new, sha, hashes

TRAINING_REVISION = "e9ef9071453a2d4021470d0586e315463868b5bb"
CHECKPOINT = "a97a0e69bb7ec7eb1e0d4bd112dbd3eedc5c9da8f27daf7a9d88b08b52f530fb"
RECEIPT = "RECOVERED_EXPORT.json"
CODE_FILES = (
    "research/code/recovery.py", "research/code/workflow_evidence.py",
    "research/code/stability_metrics.py", "research/code/runner.py",
    "research/code/model_adapter.py", "research/code/data.py",
    "research/code/determinism.py", "research/code/constants.py",
    "research/code/alignment.py", "research/tools/recover_seed401_exports.py",
)


def code_hashes(root):
    return {name: sha(root/name) for name in CODE_FILES}


def verify_original(root, original, expected_hashes):
    require(original.resolve() == (root/"research/runs/stability_v1/K5/seed_401").resolve(),
            "Only authorized original K5/seed401 path may be recovered")
    require(not (original/".RUNNING").exists() and not (original/"COMPLETED.json").exists(), "Original is active/completed")
    require(hashes(original) == expected_hashes, "Original evidence changed")
    config = read(original/"configuration.json")
    require((config["K"], config["seed"], config["implementation_git_commit"]) ==
            (5,401,TRAINING_REVISION), "Unauthorized recovery identity/revision")
    failure = read(original/"FAILED.json")
    require(failure["status"] == "FAILED" and failure["error_type"] == "StabilityInputError"
            and failure["error"] == "An archetype has no coefficient mass", "Not authorized export failure")
    require(sha(original/"final.weights.h5") == CHECKPOINT, "Wrong final checkpoint")


def verify_receipt(root, recovered):
    """Hash/identity gate, followed by full numerical validation before workflow skip."""
    require(not recovered.resolve().is_relative_to(root.resolve()), "Recovery must be outside original repository")
    record = read(recovered/RECEIPT)
    require(record["schema"] == "seed401_export_recovery_v1" and record["decision"] == "D-019"
            and record["status"] == "VERIFIED_RECOVERED_EXPORT" and record["optimizer_steps"] == 0,
            "Invalid recovery receipt")
    require(not (recovered/"COMPLETED.json").exists() and not (recovered/".RUNNING").exists(), "Recovery marker conflict")
    require(record["training_revision"] == TRAINING_REVISION and record["checkpoint_sha256"] == CHECKPOINT,
            "Recovery provenance mismatch")
    require(record["code_hashes"] == code_hashes(root), "Recovery code differs; explicit re-audit required")
    original = Path(record["original_run"])
    verify_original(root, original, record["original_hashes"])
    actual = hashes(recovered)
    require(set(actual) == set(record["recovered_hashes"]) | {RECEIPT}, "Recovery file set incomplete/changed")
    require({k:v for k,v in actual.items() if k != RECEIPT} == record["recovered_hashes"], "Recovery checksum mismatch")
    for name, digest in record["original_hashes"].items():
        require(actual.get(name) == digest, "Original snapshot not exact")
    for item in record["external_evidence"]:
        require(sha(Path(item["path"])) == item["sha256"], "Original external evidence changed")
    report = read(recovered/"recovery_validation.json")
    require(report["verdict"] == "PASS_WITH_LIMITATIONS" and (report["K"],report["seed"]) == (5,401), "Recovery validation missing/failed")
    equivalence = read(recovered/"inference_equivalence.json")
    require(equivalence["passed"] is True and equivalence["optimizer_steps"] == 0
            and equivalence["checkpoint_sha256"] == CHECKPOINT
            and set(equivalence["splits"]) == {"validation","test"}
            and equivalence["decoded_max_absolute_difference"] == 0,
            "Recovery inference verification incomplete")
    for part in equivalence["splits"].values():
        require(part == dict(rows=1002, coefficients_exact=True, latent_means_exact=True,
                            predictions_exact=True, reconstruction_mse_equivalent=True), "Recovery inference mismatch")
    return record


def require_analysis_run(root, run):
    if (run/RECEIPT).exists():
        verify_receipt(root, run)
    else:
        require((run/"COMPLETED.json").is_file() and not (run/"FAILED.json").exists()
                and not (run/".RUNNING").exists(), f"Not completed/verified recovery: {run}")
    require(read(run/"configuration.json")["status"] == "PRIMARY RUN", "Not primary")


def registered_path(root, state, k, seed, original):
    registration = state/"recoveries"/f"K{k}_seed{seed}.json"
    if not registration.exists():
        return original
    require((k,seed) == (5,401), "Unauthorized recovery registration")
    entry = read(registration)
    recovered = Path(entry["recovered_run"]).resolve()
    require(recovered.is_relative_to(state.resolve()) and recovered != original.resolve(), "Recovery registration scope")
    require(sha(recovered/RECEIPT) == entry["receipt_sha256"], "Registered receipt changed")
    record = verify_receipt(root, recovered)
    require(Path(record["original_run"]).resolve() == original.resolve(), "Registered original mismatch")
    return recovered


def inference_equivalence(root, original, model, frames):
    """Exact coefficients/latents, predictions; tolerant CSV per-image MSE replay."""
    import numpy as np
    import pandas as pd
    from .data import make_tf_dataset
    from .runner import _collect_outputs
    weight_copies = [w.numpy().copy() for w in model.weights]
    require(len(weight_copies) == 30 and all(np.isfinite(w).all() for w in weight_copies), "Invalid checkpoint tensors")
    decoded, _ = model.decode(model.z_fixed, training=False)
    old = np.load(original/"decoded_archetypes_raw.npy", allow_pickle=False)
    np.testing.assert_array_equal(decoded.numpy(), old)
    result = dict(passed=True, optimizer_steps=0, checkpoint_sha256=sha(original/"final.weights.h5"),
                  decoded_max_absolute_difference=float(abs(decoded.numpy()-old).max()), splits={})
    for split in ["validation", "test"]:
        outputs = _collect_outputs(model, make_tf_dataset(frames[split],128,32,shuffle=False,seed=None))
        for key in ["coefficients", "latent_means"]:
            np.testing.assert_array_equal(outputs[key], np.load(original/f"{split}_{key}.npy", allow_pickle=False))
        mapping = pd.read_csv(original/f"{split}_mapping.csv")
        np.testing.assert_array_equal(outputs["predictions"], mapping.predicted_label_index)
        np.testing.assert_array_equal(outputs["labels"], mapping.true_label_index)
        np.testing.assert_allclose(outputs["reconstruction_mse_per_pixel"], mapping.reconstruction_mse_per_pixel, rtol=1e-6,atol=1e-8)
        result["splits"][split] = dict(rows=len(mapping),coefficients_exact=True,latent_means_exact=True,
                                     predictions_exact=True,reconstruction_mse_equivalent=True)
    for before, after in zip(weight_copies,model.weights):
        np.testing.assert_array_equal(before,after.numpy())
    return result


def recover(root, state, destination):
    """Explicit one-time recovery, authorized on uncommitted code with source hashes."""
    import numpy as np
    import pandas as pd
    from .workflow_evidence import validate_run, archive_unit
    from .workflow import WorkflowLock, processes, competing, resources
    from .determinism import configure_determinism
    from .model_adapter import build_model
    from .data import load_journal_v1
    from .stability_metrics import semantic_class_rows, semantic_distribution_details, degeneration_details

    root, state, destination = root.resolve(), state.resolve(), destination.resolve()
    require(not state.is_relative_to(root) and destination.is_relative_to(state) and destination != state, "Recovery must be separate external child")
    require(not destination.exists(), "Recovery output already exists; never overwrite")
    registration = state/"recoveries/K5_seed401.json"
    require(not registration.exists(), "Recovery already registered")
    require(os.environ.get("PYTHONHASHSEED") == "401" and os.environ.get("TF_DETERMINISTIC_OPS") == "1", "Set seed401 determinism environment before Python")
    rows = processes()
    training = [r for r in competing(rows) if any(token in (r.get("CommandLine") or "").lower()
                for token in ["run_stability", "stability_workflow.py", "recover_seed401", "train.py", "benchmark"])]
    require(not training, "Competing training/workflow/recovery process; no recovery started")
    require(not list((root/"research/runs/stability_v1").rglob(".RUNNING")), "Active/interrupted training marker")
    resource = resources(root)
    require(resource["available_ram_bytes"] >= 1024**3 and resource["disk_free_bytes"] >= 2*1024**3, "Insufficient recovery headroom")
    original = root/"research/runs/stability_v1/K5/seed_401"
    original_hashes = hashes(original)
    verify_original(root,original,original_hashes)
    # Compare against the independently preserved pre-recovery audit, not a new seal alone.
    ledger = root.parent/"seed401_diagnosis_20260914/original_inventory.json"
    previous = read(ledger)
    require({key.removeprefix("run/"):value["sha256"] for key,value in previous.items() if key.startswith("run/")} == original_hashes, "Original differs from independent diagnosis archive")
    evidence = [dict(path=value["original"],sha256=value["sha256"]) for key,value in previous.items()
                if key.startswith(("training_log/", "session/"))]
    for item in evidence: require(sha(Path(item["path"])) == item["sha256"], "Diagnostic external evidence changed")
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    source_status = subprocess.check_output(["git","status","--short","--untracked-files=all"],cwd=root,text=True)
    code = code_hashes(root)
    lock = WorkflowLock(state,head,rows); lock.acquire()
    try:
        destination.mkdir(parents=True,exist_ok=False)
        for name in original_hashes:
            target = destination/name; target.parent.mkdir(parents=True,exist_ok=True)
            with (original/name).open("rb") as source, target.open("xb") as out:
                shutil.copyfileobj(source,out)
        for name in CODE_FILES:
            target = destination/"recovery_code"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with (root/name).open("rb") as source, target.open("xb") as out:
                shutil.copyfileobj(source,out)
        started = datetime.now(timezone.utc).isoformat()
        config = read(original/"configuration.json")
        configure_determinism(401)
        import tensorflow as tf
        _, model, _ = build_model(root,5)
        model(tf.zeros((1,128,128,3),tf.float32),training=False)
        model.load_weights(str(original/"final.weights.h5"))
        splits = root/"research/splits/journal_v1"
        image_dirs = [Path("C:/Users/UserPc/Downloads/archive")/f"HAM10000_images_part_{i}" for i in (1,2)]
        frames = load_journal_v1(splits/"train.csv",splits/"validation.csv",splits/"test.csv",image_dirs)
        equivalence = inference_equivalence(root,original,model,frames)
        write_new(destination/"inference_equivalence.json",equivalence)
        a = np.load(original/"test_coefficients.npy",allow_pickle=False)
        pd.DataFrame(semantic_class_rows(a,frames["test"].dx)).to_csv(destination/"test_archetype_class_distribution_raw.csv",mode="x",index=False,lineterminator="\n")
        write_new(destination/"test_semantic_status.json",semantic_distribution_details(a,frames["test"].dx))
        write_new(destination/"degeneration.json",{part:degeneration_details(np.load(original/f"{part}_coefficients.npy")) for part in ["validation","test"]})
        write_new(destination/"warnings.json",config["determinism"]["warnings"])
        provenance = dict(original_run=str(original),original_hashes=original_hashes)
        validation = validate_run(root,destination,5,401,[TRAINING_REVISION],_recovery=provenance)
        write_new(destination/"recovery_validation.json",validation)
        write_new(destination/"recovery_environment.json",dict(started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(),executable=sys.executable,tensorflow=tf.__version__,numpy=np.__version__,
            resources_before=resource,warning_provenance="warnings.json copies original configuration.determinism.warnings only",
            missing_historical_metadata=["runtime_environment.json", "complete historical warning inventory"],
            note="Recovery environment, NOT original training runtime"))
        require(code_hashes(root) == code and hashes(original) == original_hashes, "Source/original changed during recovery")
        write_new(destination/RECEIPT,dict(schema="seed401_export_recovery_v1",decision="D-019",status="VERIFIED_RECOVERED_EXPORT",
            optimizer_steps=0,training_revision=TRAINING_REVISION,recovery_revision=head,recovery_git_status=source_status,
            recovery_revision_note="HEAD plus exact code hashes; uncommitted recovery code is explicitly disclosed",
            code_hashes=code,checkpoint_sha256=CHECKPOINT,external_evidence=evidence,**provenance,recovered_hashes=hashes(destination)))
        verify_receipt(root,destination)
        archive = archive_unit(state/"archives","recovered_K5_seed401",dict(original=original,recovered=destination,
            training_log=Path(next(x["path"] for x in evidence if x["path"].endswith(".log") and "training_logs" in x["path"])),
            session=state/"sessions/20260914T014247665284Z"),hashes(destination))
        registration.parent.mkdir(parents=True,exist_ok=True)
        write_new(registration,dict(recovered_run=str(destination),receipt_sha256=sha(destination/RECEIPT),archive=str(archive),archive_sha256=sha(archive)))
        return dict(status="VERIFIED_RECOVERED_EXPORT",recovered_run=str(destination),registration=str(registration),archive=str(archive))
    finally:
        lock.release()
