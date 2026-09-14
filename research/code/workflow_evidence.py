"""Read-only validation and non-overwriting derivatives for the frozen workflow.

Generalizes the September seed113/211 audit; never calls train_step.
Heavy numerical imports are local so status/preflight do not load TensorFlow.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import shutil
import subprocess
import zipfile
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hashes(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), f"Symlink evidence refused: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def validate_run(root, run, k, seed, approved_heads):
    import numpy as np
    import pandas as pd
    from .constants import (BASELINE_SOURCE, BASELINE_SOURCE_SHA256, BASELINE_COMMIT,
                            MANIFEST_SHA256, LOSS_WEIGHTS, EXPECTED_CLASSES)
    from .alignment import validate_coefficients
    from .model_adapter import build_model
    from .stability_metrics import nearest_real_top_n, coefficient_weighted_class_distribution

    before = hashes(run)
    require(not (run / ".RUNNING").exists() and not (run / "FAILED.json").exists(), "Active/failed run")
    completion = read(run / "COMPLETED.json")
    require(completion["status"] == "COMPLETED", "Not completed")
    require(completion["baseline_source_sha256_before"] == completion["baseline_source_sha256_after"]
            == sha(root / BASELINE_SOURCE) == BASELINE_SOURCE_SHA256, "Protected source mismatch")
    config = read(run / "configuration.json")
    expected = dict(K=k, latent_dimension=k-1, seed=seed, epochs=50, batch_size=32,
                    image_shape=[128,128,3], learning_rate=.001, early_stopping=False,
                    checkpoint_rule="final epoch", smoke_test=False, status="PRIMARY RUN",
                    optimizer="tf.keras.optimizers.Adam", loss_weights=LOSS_WEIGHTS,
                    protocol="archetype_stability_v1", conference_baseline_commit=BASELINE_COMMIT,
                    raw_unaligned_archetype_numbering=True, implementation_git_status_before_run="")
    for key, value in expected.items():
        require(config.get(key) == value, f"Configuration mismatch: {key}")
    revision = config["implementation_git_commit"]
    require(revision in approved_heads, f"Unapproved execution revision: {revision}")
    subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=root, check=True, capture_output=True)
    # Scientific implementation must still match the approved run's Git objects.
    for file in [BASELINE_SOURCE, "research/code/constants.py", "research/code/model_adapter.py",
                 "research/code/data.py", "research/code/determinism.py"]:
        result = subprocess.run(["git", "diff", "--quiet", revision, "HEAD", "--", file], cwd=root)
        require(result.returncode == 0, f"Scientific source differs from execution: {file}")
    det = config["determinism"]
    for name in ["seed", "python_random_seed", "numpy_seed", "tensorflow_seed", "keras_seed",
                 "model_initialization_seed", "posterior_sampling_seed", "dataset_shuffle_seed"]:
        require(det[name] == seed, f"Seed mismatch: {name}")
    require(det["deterministic_ops_enabled"], "Deterministic operations not enabled")
    require(config["process_determinism_environment"] ==
            {"PYTHONHASHSEED":str(seed), "TF_DETERMINISTIC_OPS":"1"}, "Process seed mismatch")
    manifests = {}
    for part, count in [("train",8011),("validation",1002),("test",1002)]:
        path = root / "research/splits/journal_v1" / (part + ".csv")
        require(sha(path) == MANIFEST_SHA256[path.name] == config["manifests"][part]["sha256"], "Manifest hash")
        require(config["manifests"][part]["full_rows"] == config["manifests"][part]["used_rows"] == count, "Subset run")
        manifests[part] = pd.read_csv(path)
    inv = read(run / "artifact_inventory.json")["artifacts"]
    require(len({r["path"] for r in inv}) == len(inv), "Duplicate inventory path")
    require(set(before) == {r["path"] for r in inv} | {"artifact_inventory.json", "COMPLETED.json"}, "Inventory file set")
    for item in inv:
        path = (run / item["path"]).resolve()
        require(path.is_relative_to(run.resolve()), "Unsafe inventory path")
        require(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], f"Inventory mismatch {path}")
    history = pd.read_csv(run / "training_history.csv")
    require(history.epoch.tolist() == list(range(1,51)), "Epoch sequence not 1..50")
    require(np.isfinite(history.to_numpy()).all(), "Nonfinite history")
    metrics = read(run / "split_metrics.json")
    require(set(metrics) == {"train","validation","test"}, "Split metric set")
    for part in ["train","validation"]:
        expected_total = history[part+"_reconstruction_sum"] + history[part+"_classification_ce"] + .01*history[part+"_kl"] + history[part+"_archetype_simplex"]
        np.testing.assert_allclose(history[part+"_total"], expected_total, rtol=1e-6, atol=1e-4)
    for values in metrics.values():
        require(np.isfinite(list(values.values())).all(), "Nonfinite split metrics")
        np.testing.assert_allclose(values["total"], values["reconstruction_sum"] + values["classification_ce"] + .01*values["kl"] + values["archetype_simplex"], rtol=1e-6, atol=1e-4)
    mappings, arrays = {}, {}
    for part in ["validation", "test"]:
        mapping = pd.read_csv(run / f"{part}_mapping.csv")
        for col in ["image_id","lesion_id","dx","source_partition","fold"]:
            require(mapping[col].tolist() == manifests[part][col].tolist(), f"{part} mapping/order: {col}")
        require(mapping.true_label_index.tolist() == [EXPECTED_CLASSES.index(x) for x in mapping.dx], "Label indices")
        require(mapping.predicted_label_index.between(0,6).all(), "Prediction range")
        require(mapping.predicted_dx.tolist() == [EXPECTED_CLASSES[x] for x in mapping.predicted_label_index], "Prediction labels")
        a = np.load(run / f"{part}_coefficients.npy", allow_pickle=False)
        z = np.load(run / f"{part}_latent_means.npy", allow_pickle=False)
        validate_coefficients(part, a)
        require(a.shape == (1002,k) and z.shape == (1002,k-1) and np.isfinite(z).all(), "Array shape/finite")
        mse = mapping.reconstruction_mse_per_pixel.to_numpy()
        require(np.isfinite(mse).all() and (mse >= 0).all(), "Per-image errors")
        np.testing.assert_allclose(mse.mean(), metrics[part]["reconstruction_mse_per_pixel"], rtol=1e-6)
        np.testing.assert_allclose(np.mean([mse[i:i+32].mean()*49152 for i in range(0,len(mse),32)]), metrics[part]["reconstruction_sum"], rtol=1e-6)
        np.testing.assert_allclose((mapping.true_label_index == mapping.predicted_label_index).mean(), metrics[part]["accuracy"], atol=1e-12)
        mappings[part], arrays[part] = mapping, (a,z)
    from .determinism import configure_determinism
    configure_determinism(seed)
    import tensorflow as tf
    _, model, _ = build_model(root, k)
    model(tf.zeros((1,128,128,3)), training=False)
    model.load_weights(str(run / "final.weights.h5"))
    require(len(model.weights) == 30 and all(np.isfinite(w.numpy()).all() for w in model.weights), "Checkpoint tensors")
    require(read(run / "checkpoint_reload.json") == dict(loaded=True, passed=True, weight_tensor_count=30, maximum_absolute_weight_difference=0.0), "Recorded reload")
    for a,z in arrays.values():
        np.testing.assert_allclose(z, a @ model.z_fixed.numpy(), rtol=1e-6, atol=1e-6)
    decoded = np.load(run / "decoded_archetypes_raw.npy", allow_pickle=False)
    require(decoded.shape == (k,128,128,3) and np.isfinite(decoded).all() and decoded.min() >= 0 and decoded.max() <= 1, "Decoded arrays")
    fresh, _ = model.decode(model.z_fixed, training=False)
    np.testing.assert_allclose(fresh.numpy(), decoded, rtol=1e-5, atol=1e-6)
    for i in range(k):
        png = tf.io.decode_png(tf.io.read_file(str(run / f"decoded_archetype_raw_A{i+1}.png")), channels=3).numpy()
        np.testing.assert_array_equal(png, tf.image.convert_image_dtype(decoded[i], tf.uint8, saturate=True).numpy())
    nearest = pd.read_csv(run / "test_nearest_real_top10_raw.csv")
    calculated = pd.DataFrame([r for group in nearest_real_top_n(arrays["test"][1], model.z_fixed.numpy(), mappings["test"].image_id, mappings["test"].dx) for r in group])
    pd.testing.assert_frame_equal(nearest, calculated, check_dtype=False, atol=1e-8, rtol=1e-6)
    table = pd.read_csv(run / "test_archetype_class_distribution_raw.csv")
    require(len(table)==k*7 and set(zip(table.raw_archetype,table.dx)) == {(i,c) for i in range(1,k+1) for c in EXPECTED_CLASSES}, "Class table keys")
    dist = coefficient_weighted_class_distribution(arrays["test"][0], mappings["test"].dx)
    for row in table.itertuples():
        prob = dist[row.raw_archetype-1, EXPECTED_CLASSES.index(row.dx)]
        prevalence = float((mappings["test"].dx == row.dx).mean())
        np.testing.assert_allclose([row.coefficient_weighted_probability,row.test_dx_prevalence,row.enrichment_ratio], [prob,prevalence,prob/prevalence], rtol=1e-6)
    if (run / "progress.jsonl").exists():
        events = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
        require([e["epoch"] for e in events if e["event"] == "epoch_end"] == list(range(1,51)), "Progress epoch sequence")
    require(hashes(run) == before, "Run modified during validation")
    return dict(verdict="PASS_WITH_LIMITATIONS", K=k, seed=seed, run=str(run), execution_revision=revision,
                hashes=before, metrics=metrics, decoded_reload_max_difference=float(np.max(np.abs(fresh.numpy()-decoded))),
                warnings=read(run/"warnings.json"), limitation="Artifact/reload consistency, not trajectory replay or clinical validity")


def verify_pair(root, pair, reference, candidate, k, seed):
    """Recompute the existing protocol from saved inputs; never refit on test."""
    import numpy as np
    import pandas as pd
    from .alignment import align_coefficients
    from .model_adapter import load_baseline_module
    from .stability_metrics import (coefficient_stability, paired_lesion_coefficients,
        lesion_bootstrap_coefficient_stability, semantic_stability, nearest_real_top_n,
        compare_nearest_real, visual_stability)
    before = hashes(pair)
    input_hashes = {"reference":hashes(reference), "candidate":hashes(candidate)}
    p = read(pair/"alignment/permutation.json")
    require(Path(p["reference_run"]).resolve()==reference.resolve() and Path(p["candidate_run"]).resolve()==candidate.resolve(), "Pair input paths")
    require((p["K"],p["reference_seed"],p["candidate_seed"],p["matching_partition"],p["test_used_for_matching"]) == (k,113,seed,"validation",False), "Pair protocol")
    mappings, values = {}, {}
    for part in ["validation","test"]:
        mappings[part] = [pd.read_csv(run/f"{part}_mapping.csv") for run in [reference,candidate]]
        pd.testing.assert_frame_equal(mappings[part][0][["image_id","lesion_id","dx"]],mappings[part][1][["image_id","lesion_id","dx"]])
        values[part] = [np.load(run/f"{part}_coefficients.npy",allow_pickle=False) for run in [reference,candidate]]
    aligned = align_coefficients(*values["validation"])
    permutation = aligned.candidate_column_for_reference
    require(p["candidate_column_for_reference"] == permutation.tolist(), "Permutation mismatch")
    for key,value in aligned.summary().items():
        if isinstance(value,list): np.testing.assert_allclose(p[key],value,rtol=1e-8,atol=1e-12)
        else: require(p[key]==value, f"Alignment metadata {key}")
    np.testing.assert_allclose(pd.read_csv(pair/"alignment/cost_matrix.csv",header=None),aligned.cost_matrix,atol=1e-12)
    np.testing.assert_array_equal(pd.read_csv(pair/"alignment/cost_method_matrix.csv",header=None),aligned.cost_method_matrix)
    for part in ["validation","test"]:
        np.testing.assert_array_equal(np.load(pair/f"alignment/candidate_{part}_coefficients_aligned.npy"),values[part][1][:,permutation])
    ref, cand = values["test"][0], values["test"][1][:,permutation]
    rm, cm = mappings["test"]
    coefficient = coefficient_stability(ref,rm,cand,cm)
    rl,cl,lm = paired_lesion_coefficients(ref,rm,cand,cm)
    require(len(lm)==748, "Lesion aggregation count")
    boot = lesion_bootstrap_coefficient_stability(rl,cl)
    semantic = semantic_stability(ref,rm.dx,cand,cm.dx)
    baseline = load_baseline_module(root)
    vertices = baseline.create_z_fixed(k-1)
    nearest = [nearest_real_top_n(np.load(run/"test_latent_means.npy"),vertices,m.image_id,m.dx) for run,m in [(reference,rm),(candidate,cm)]]
    nearest = compare_nearest_real(*nearest,permutation)
    visual = visual_stability(np.load(reference/"decoded_archetypes_raw.npy"),np.load(candidate/"decoded_archetypes_raw.npy"),permutation)
    expected = dict(K=k, reference_seed=113,candidate_seed=seed, matching_partition="validation",evaluation_partition="test",
        permutation=permutation.tolist(),coefficient={key:value for key,value in coefficient.items() if key not in {"coefficient_vector_l1_by_lesion","dominant_agreement_by_lesion","lesion_mapping"}},
        lesion_bootstrap=boot,semantic={"mean_jensen_shannon_divergence_base2":semantic["mean_jensen_shannon_divergence_base2"]},nearest_real=nearest,visual=visual)
    def compare(a,b):
        if isinstance(b,dict):
            require(set(a)==set(b),"Summary keys")
            for key in b: compare(a[key],b[key])
        elif isinstance(b,list):
            require(len(a)==len(b),"Summary length")
            for x,y in zip(a,b): compare(x,y)
        elif b is None or isinstance(b,(str,bool)):
            require(a==b,"Undefined/string/bool mismatch")
        else: np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8)
    compare(read(pair/"test/stability_summary.json"),expected)
    lesion_rows = lm.copy()
    lesion_rows["coefficient_vector_l1"] = coefficient["coefficient_vector_l1_by_lesion"]
    lesion_rows["dominant_archetype_agreement"] = coefficient["dominant_agreement_by_lesion"]
    for name,frame in [("coefficient_stability_by_lesion.csv",lesion_rows),("nearest_real_stability.csv",pd.DataFrame(nearest)),("visual_stability.csv",pd.DataFrame(visual))]:
        pd.testing.assert_frame_equal(pd.read_csv(pair/"test"/name),frame,check_dtype=False,rtol=1e-6,atol=1e-8)
    from .constants import EXPECTED_CLASSES
    semantic_rows = [dict(matched_archetype=i+1,dx=c,reference_probability=semantic["reference_distribution"][i,j],candidate_probability=semantic["candidate_distribution"][i,j],jensen_shannon_divergence_base2=semantic["jensen_shannon_divergence_base2"][i]) for i in range(k) for j,c in enumerate(EXPECTED_CLASSES)]
    pd.testing.assert_frame_equal(pd.read_csv(pair/"test/semantic_stability.csv"),pd.DataFrame(semantic_rows),rtol=1e-6,atol=1e-8)
    legacy = pair/"presentation/derived_sha256.json"
    if legacy.exists():
        for name,digest in read(legacy).items():
            path = (pair/name).resolve()
            require(path.is_relative_to(pair.resolve()) and sha(path)==digest,"Legacy derivative hash")
    require(hashes(pair)==before and hashes(reference)==input_hashes["reference"] and hashes(candidate)==input_hashes["candidate"],"Evidence changed")
    return dict(verdict="PASS", K=k, seed=seed, pair=str(pair), inputs=input_hashes, hashes=before, summary=expected)


def archive_unit(destination, label, sources, identity):
    """One verified content-addressed ZIP per immutable run/pair, not archives of archives."""
    destination.mkdir(parents=True,exist_ok=True)
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    target = destination/f"{label}_{key}.zip"
    if target.exists():
        with zipfile.ZipFile(target) as z:
            manifest = json.loads(z.read("MANIFEST.json"))
            require(manifest["identity"]==identity and z.testzip() is None,"Existing archive corrupt")
            require(set(z.namelist())=={e["member"] for e in manifest["files"]}|{"MANIFEST.json"},"Archive member set changed")
            for entry in manifest["files"]:
                require(hashlib.sha256(z.read(entry["member"])).hexdigest()==entry["sha256"],"Existing archive checksum")
                # Session events may grow after their archived checkpoint. Immutable
                # run/pair/log/validation sources must still match their archived bytes.
                if not entry["member"].startswith("session/"):
                    require(Path(entry["source"]).is_file() and sha(entry["source"])==entry["sha256"],"Archived immutable source changed/missing")
        return target
    entries = []
    for label_name,source in sources.items():
        source = Path(source)
        require(source.exists(),f"Missing archive input {source}")
        for path in sorted(source.rglob("*") if source.is_dir() else [source]):
            require(not path.is_symlink(),"Symlink archive input")
            if path.is_file():
                member = label_name+"/"+(path.relative_to(source).as_posix() if source.is_dir() else path.name)
                entries.append(dict(source=str(path),member=member,bytes=path.stat().st_size,sha256=sha(path)))
    required = sum(e["bytes"] for e in entries)*2 + 2*1024**3
    require(shutil.disk_usage(destination).free > required,"Insufficient archive disk headroom")
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for e in entries: z.write(e["source"],e["member"])
        z.writestr("MANIFEST.json",json.dumps(dict(identity=identity,files=entries,same_disk_not_disaster_backup=True),indent=2))
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None,"Archive CRC")
        for e in entries:
            require(hashlib.sha256(z.read(e["member"])).hexdigest()==e["sha256"]==sha(e["source"]),"Archive/source checksum")
    return target


def figure(pair, reference, candidate, destination):
    """SVG layout embedding original PNG bytes, not generated imagery."""
    p = read(pair/"alignment/permutation.json")["candidate_column_for_reference"]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{90+len(p)*165}">',
             '<rect width="100%" height="100%" fill="white"/><g font-family="Arial" fill="black">',
             '<text x="20" y="25">Validation-aligned decoded archetypes (unaltered PNGs)</text>']
    for i,j in enumerate(p):
        for run,raw,x,title in [(reference,i+1,20,"reference113"),(candidate,j+1,310,"candidate"+str(read(candidate/"configuration.json")["seed"]))]:
            data = base64.b64encode((run/f"decoded_archetype_raw_A{raw}.png").read_bytes()).decode()
            y = 55 + i*165
            lines += [f'<text x="{x}" y="{y}">Matched A{i+1}: {title} raw A{raw}</text>',f'<image x="{x}" y="{y+10}" width="128" height="128" href="data:image/png;base64,{data}"/>']
    lines.append('</g></svg>')
    with destination.open("x",encoding="utf-8") as f: f.write("\n".join(lines))


def aggregate(destination, run_reports, pair_reports):
    """Separate K summaries; observed denominators, never imputed correlations/seed CIs."""
    import numpy as np
    from .stability_metrics import summarize_seed_comparisons
    for r in run_reports:
        require(r["verdict"]=="PASS_WITH_LIMITATIONS" and hashes(Path(r["run"]))==r["hashes"],"Unverified/changed run at aggregation")
    run_lookup={(r["K"],r["seed"]):r for r in run_reports}
    for p in pair_reports:
        require(p["verdict"]=="PASS" and hashes(Path(p["pair"]))==p["hashes"],"Unverified/changed pair at aggregation")
        require((p["K"],113) in run_lookup and (p["K"],p["seed"]) in run_lookup,"Pair lacks verified reference/candidate")
        require(p["inputs"]==dict(reference=run_lookup[p["K"],113]["hashes"],candidate=run_lookup[p["K"],p["seed"]]["hashes"]),"Pair input hashes differ from verified runs")
    destination.mkdir(parents=True,exist_ok=False)
    raw = []
    for report in pair_reports:
        s = report["summary"]; k=s["K"]; seed=s["candidate_seed"]
        def add(metric,value,archetype=0,unit="pair",**extra):
            require(value is None or (isinstance(value,(int,float)) and math.isfinite(value)),"Invalid summary number")
            raw.append(dict(K=k,seed=seed,reference_seed=113,archetype=archetype,metric=metric,value=value,unit=unit,**extra))
        for name in ["coefficient_vector_l1_mean","dominant_archetype_agreement"]:
            ci=s["lesion_bootstrap"]["metrics"][name]
            add(name,s["coefficient"][name],unit="748 lesion means",lesion_bootstrap_lower=ci["lower"],lesion_bootstrap_upper=ci["upper"])
        add("mean_semantic_js_base2",s["semantic"]["mean_jensen_shannon_divergence_base2"],unit="image-coefficient-weighted distributions")
        semantic = read_csv_rows(Path(report["pair"])/"test/semantic_stability.csv")
        for i in range(k):
            c=s["coefficient"]["per_archetype"][i]
            add("lesion_pearson",c["pearson_correlation"],i+1,"748 lesion means",correlation_defined=c["correlation_defined"])
            add("lesion_mae",c["mae"],i+1,"748 lesion means")
            add("semantic_js_base2",float(next(r for r in semantic if int(r["matched_archetype"])==i+1)["jensen_shannon_divergence_base2"]),i+1,"1002 image coefficient weights")
            for key in ["image_id_jaccard","diagnosis_jensen_shannon_divergence_base2"]: add(key,s["nearest_real"][i][key],i+1,"top10 test images")
            for key in ["rgb_ssim","pixel_mse"]: add(key,s["visual"][i][key],i+1,"decoded RGB128 image")
    stats=[]
    for key in sorted({(r["K"],r["archetype"],r["metric"]) for r in raw}):
        rows=[r for r in raw if (r["K"],r["archetype"],r["metric"])==key]
        vals=[r["value"] for r in rows if r["value"] is not None]
        if len(vals)==4:
            st=summarize_seed_comparisons(vals)
            st={n:st[n] for n in ["mean","sample_standard_deviation","minimum","maximum"]}
        else:
            st=dict(mean=float(np.mean(vals)) if vals else None,sample_standard_deviation=float(np.std(vals,ddof=1)) if len(vals)>1 else None,minimum=min(vals) if vals else None,maximum=max(vals) if vals else None)
        stats.append(dict(K=key[0],archetype=key[1],metric=key[2],expected_pairs=4,observed_pairs=len(rows),defined_values=len(vals),undefined_values=len(rows)-len(vals),scope="descriptive observed values; no seed CI",**st))
    run_keys=[(r["K"],r["seed"]) for r in run_reports]
    pair_keys=[(r["K"],r["seed"]) for r in pair_reports]
    require(len(set(run_keys))==len(run_keys) and len(set(pair_keys))==len(pair_keys),"Duplicate summary inputs")
    expected_runs={(k,s) for k in (5,7) for s in (113,211,307,401,503)}
    expected_pairs={(k,s) for k,s in expected_runs if s!=113}
    require(set(run_keys)<=expected_runs and set(pair_keys)<=expected_pairs,"Out-of-protocol summary input")
    complete=set(run_keys)==expected_runs and set(pair_keys)==expected_pairs
    result=dict(status="COMPLETE" if complete else "INCOMPLETE",verified_runs=[dict(K=r["K"],seed=r["seed"],run=r["run"],execution_revision=r["execution_revision"],hashes=r["hashes"]) for r in run_reports],
                verified_pairs=pair_reports, raw_metrics=raw, descriptive_statistics=stats,
                note="K5 and K7 separate; shared-reference variation is not independent experimental uncertainty. Lesion bootstrap only L1/agreement, 1000 draws seed2026. No stability threshold or clinical claim.")
    write_new(destination/"summary.json",result)
    csv_rows(destination/"raw_metrics.csv",raw)
    csv_rows(destination/"descriptive_statistics.csv",stats)
    text=[f'# Stability v1 technical findings: {result["status"]}',f'\nVerified runs: {len(run_reports)}/10; comparisons: {len(pair_reports)}/8.', '\n'+result["note"], '\n## Pair-level coefficient findings\n', '| K | Candidate | Lesions | L1 | Dominant agreement |','|---|---|---|---|---|']
    for r in pair_reports:
        s=r["summary"];c=s["coefficient"]
        text.append(f'| {s["K"]} | {s["candidate_seed"]} | {c["lesions"]} | {c["coefficient_vector_l1_mean"]:.6f} | {c["dominant_archetype_agreement"]:.6f} |')
        figure(Path(r["pair"]),Path(r["summary_input_reference"]),Path(r["summary_input_candidate"]),destination/f'K{s["K"]}_seed{s["candidate_seed"]}.svg')
    text += ['\nAll per-pair/per-archetype coefficient, semantic, nearest-image and visual values are in raw_metrics.csv; per-K statistics and defined denominators are in descriptive_statistics.csv. Null means undefined, never zero. Observed-only statistics in incomplete summaries are not the final four-comparison result. Lesion bootstrap bounds are per-pair uncertainty, not across-run confidence intervals.', '\nNo metric-driven tuning or stable/unstable threshold. Strong or weak metrics do not determine technical pass/fail. Inspect each K separately; complete execution is not evidence of clinical validity or overall reproducibility.']
    with (destination/"FINDINGS.md").open("x",encoding="utf-8") as f:f.write("\n".join(text)+"\n")
    write_new(destination/"summary_files_sha256.json",hashes(destination))
    return result


def read_csv_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def csv_rows(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({key:("undefined" if value is None else value) for key,value in row.items()} for row in rows)
