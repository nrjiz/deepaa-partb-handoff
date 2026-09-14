"""Orchestration fixtures only: no primary runner, TensorFlow or training."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.code import workflow as module
from research.code.workflow import Workflow, WorkflowLock, MATRIX, classify, competing, release_stale
from research.code.workflow_evidence import archive_unit, aggregate, write_new


@pytest.fixture
def queue(tmp_path,monkeypatch):
    root=tmp_path/"repo";root.mkdir()
    w=Workflow(root,tmp_path/"external","a"*40)
    w.session=w.state/"sessions"/"test";w.session.mkdir(parents=True)
    calls=[]
    monkeypatch.setattr(w,"guard",lambda **kw:{})
    monkeypatch.setattr(w,"emit",lambda *a,**kw:None)
    def launch(k,s):
        calls.append(("launch",k,s))
        path=w.run_path(k,s);path.mkdir(parents=True)
        write_new(path/"COMPLETED.json",{})
    def validate(k,s):
        calls.append(("validate",k,s))
        return dict(K=k,seed=s,execution_revision="old_valid",run=str(w.run_path(k,s)),hashes={})
    def pair(k,s,create):
        calls.append(("pair",k,s))
        return dict(K=k,seed=s) if create else None
    monkeypatch.setattr(w,"launch",launch);monkeypatch.setattr(w,"validate",validate)
    monkeypatch.setattr(w,"pair",pair);monkeypatch.setattr(w,"backup_run",lambda r:None)
    monkeypatch.setattr(module,"aggregate",lambda path,r,p:{"status":"COMPLETE" if len(r)==10 and len(p)==8 else "INCOMPLETE"})
    monkeypatch.setattr(module,"hashes",lambda path:{})
    monkeypatch.setattr(module,"archive_unit",lambda *a:tmp_path/"archive.zip")
    return w,calls


def completed(w,k,s):
    p=w.run_path(k,s);p.mkdir(parents=True);write_new(p/"COMPLETED.json",{})


def test_skip_verified_and_order(queue):
    w,calls=queue
    for seed in [113,211,307]:completed(w,5,seed)
    w.work(True)
    assert [c[1:] for c in calls if c[0]=="launch"]==list(MATRIX[3:])
    assert [c[1:] for c in calls if c[0]=="validate"]==list(MATRIX)
    assert len([c for c in calls if c[0]=="pair"])==8


def test_partial_refusal(queue):
    w,calls=queue;w.run_path(5,113).mkdir(parents=True)
    with pytest.raises(ValueError,match="PARTIAL"):w.work(True)
    assert not calls


def test_failed_validation_stops_before_next_launch(queue,monkeypatch):
    w,calls=queue;completed(w,5,113)
    def bad(*args):raise ValueError("bad checksum")
    monkeypatch.setattr(w,"validate",bad)
    with pytest.raises(ValueError,match="bad checksum"):w.work(True)
    assert not calls


def test_summarize_never_launches_missing(queue):
    w,calls=queue;completed(w,5,113)
    w.work(False)
    assert calls==[("validate",5,113)]
    result=json.loads((w.session/"result.json").read_text())
    assert result["status"]=="INCOMPLETE"


def test_native_exit_seed_and_direct_logs(tmp_path,monkeypatch):
    w=Workflow(tmp_path/"repo",tmp_path/"state","a"*40)
    w.session=tmp_path/"logs";w.session.mkdir()
    seen={}
    class Child:
        pid=123
        def __init__(self,command,**kw):seen.update(command=command,**kw)
        def wait(self):return 7
    monkeypatch.setattr(module.subprocess,"Popen",Child)
    with pytest.raises(ValueError,match="native exit 7"):
        w.invoke(["python","train"],"failure",307)
    assert seen["env"]["PYTHONHASHSEED"]=="307"
    assert seen["env"]["TF_DETERMINISTIC_OPS"]=="1"
    assert seen["stdout"].name.endswith("failure.log")
    assert seen["stderr"]==module.subprocess.STDOUT
    assert w.child is None


def test_lock_duplicate_and_stale_identity(tmp_path):
    rows=[dict(ProcessId=os.getpid(),ParentProcessId=0,CreationDate="now",Name="python.exe",CommandLine="workflow")]
    lock=WorkflowLock(tmp_path,"a"*40,rows);lock.acquire()
    with pytest.raises(FileExistsError):WorkflowLock(tmp_path,"a"*40,rows).acquire()
    with pytest.raises(ValueError,match="still alive"):release_stale(tmp_path,rows)
    lock.release()
    lock.acquire()
    release_stale(tmp_path,[])  # OS confirms no owner; preserve rather than delete.
    assert not lock.path.exists()
    assert len(list((tmp_path/"released_locks").glob("*.json")))==2


def test_competing_process_does_not_exclude_unrelated_python():
    rows=[dict(ProcessId=os.getpid(),ParentProcessId=111,CreationDate="a",Name="python.exe",CommandLine="queue"),
          dict(ProcessId=111,ParentProcessId=0,CreationDate="b",Name="python.exe",CommandLine="venv launcher"),
          dict(ProcessId=222,ParentProcessId=0,CreationDate="c",Name="python.exe",CommandLine="train")]
    assert [r["ProcessId"] for r in competing(rows)]==[222]


def test_archive_skip_verified_no_recursive_archives(tmp_path):
    source=tmp_path/"source";source.mkdir();(source/"evidence.txt").write_text("original")
    from research.code.workflow_evidence import hashes
    target=archive_unit(tmp_path/"backups","unit",dict(evidence=source),hashes(source))
    original=target.read_bytes()
    assert archive_unit(tmp_path/"backups","unit",dict(evidence=source),hashes(source))==target
    assert target.read_bytes()==original


def test_existing_pair_reuse_does_not_invoke_alignment(tmp_path,monkeypatch):
    w=Workflow(tmp_path/"repo",tmp_path/"external","a"*40)
    w.session=tmp_path/"session";w.session.mkdir()
    pair=w.pair_path(5,211);(pair/"alignment").mkdir(parents=True)
    (pair/"alignment/cost_method_matrix.csv").write_text("pearson,pearson\npearson,pearson\n")
    monkeypatch.setattr(w,"guard",lambda **kw:{})
    seen=[]
    def invoke(command,label,*args,**kwargs):
        seen.append(label)
        write_new(w.session/"pair_K5_seed211.json",dict(verdict="PASS",K=5,seed=211,hashes={},inputs={},summary={"permutation":[0,1]}))
    monkeypatch.setattr(w,"invoke",invoke)
    monkeypatch.setattr(module,"hashes",lambda path:{})
    monkeypatch.setattr(module,"archive_unit",lambda *a:tmp_path/"backup.zip")
    w.pair(5,211,True)
    assert seen==["verify_pair_K5_seed211"]


def test_corrupt_existing_pair_stops(tmp_path,monkeypatch):
    w=Workflow(tmp_path/"repo",tmp_path/"external","a"*40)
    w.session=tmp_path/"session";w.session.mkdir()
    w.pair_path(5,211).mkdir(parents=True)
    monkeypatch.setattr(w,"guard",lambda **kw:{})
    monkeypatch.setattr(w,"invoke",lambda *a,**kw:(_ for _ in ()).throw(ValueError("corrupt pair")))
    with pytest.raises(ValueError,match="corrupt pair"):w.pair(5,211,True)


def test_empty_summary_is_explicitly_incomplete(tmp_path):
    s=aggregate(tmp_path/"summary",[],[])
    assert s["status"]=="INCOMPLETE" and s["raw_metrics"]==[]
    assert (tmp_path/"summary/FINDINGS.md").exists()


def test_no_unknown_revision_rejection_for_known_old_runs(tmp_path):
    w=Workflow(tmp_path/"repo",tmp_path/"external","a"*40)
    assert "0e32c8d469be141d30e627219aeac7ee253a34fa" in w.allowed_heads(5,113)
    assert "f097f2f9e6ad72464685dacca59ba95ba273ab33" in w.allowed_heads(5,211)
    assert w.allowed_heads(7,113)==["a"*40]


def test_active_marker_wins_over_completion(tmp_path):
    write_new(tmp_path/"COMPLETED.json",{})
    (tmp_path/".RUNNING").write_text("active")
    assert classify(tmp_path)=="ACTIVE_OR_INTERRUPTED"


def test_read_only_live_viewer_does_not_block_queue():
    script=Path(module.__file__).resolve().parents[1]/"tools/stability_workflow.py"
    row=dict(ProcessId=999999,ParentProcessId=0,CreationDate="x",Name="python.exe",CommandLine=f'python -B "{script}" watch')
    assert competing([row])==[]
    row["CommandLine"]=f'python -B "{script}" finish'
    assert competing([row])==[row]


def fixture_summary_inputs(tmp_path,keys):
    from research.code.workflow_evidence import hashes
    runs=[];pairs=[]
    for k,seed in keys:
        run=tmp_path/f"run_{k}_{seed}";run.mkdir()
        write_new(run/"configuration.json",dict(seed=seed))
        for i in range(k):(run/f"decoded_archetype_raw_A{i+1}.png").write_bytes(b"fixture_image")
        runs.append(dict(verdict="PASS_WITH_LIMITATIONS",K=k,seed=seed,run=str(run),execution_revision="fixture",hashes=hashes(run)))
    by_key={(r["K"],r["seed"]):r for r in runs}
    for k,seed in keys:
        if seed==113 or (k,113) not in by_key:continue
        path=tmp_path/f"pair_{k}_{seed}";(path/"test").mkdir(parents=True);(path/"alignment").mkdir()
        write_new(path/"alignment/permutation.json",dict(candidate_column_for_reference=list(range(k))))
        (path/"test/semantic_stability.csv").write_text("matched_archetype,jensen_shannon_divergence_base2\n"+"".join(f"{i+1},0.1\n" for i in range(k)))
        summary=dict(K=k,candidate_seed=seed,coefficient=dict(lesions=748,coefficient_vector_l1_mean=.3,dominant_archetype_agreement=.5,
            per_archetype=[dict(pearson_correlation=None if seed==211 else .7,correlation_defined=seed!=211,mae=.05) for _ in range(k)]),
            semantic=dict(mean_jensen_shannon_divergence_base2=.1),lesion_bootstrap=dict(metrics={n:dict(lower=.1,upper=.9) for n in ["coefficient_vector_l1_mean","dominant_archetype_agreement"]}),
            nearest_real=[dict(image_id_jaccard=.1,diagnosis_jensen_shannon_divergence_base2=.2) for _ in range(k)],
            visual=[dict(rgb_ssim=.7,pixel_mse=.1) for _ in range(k)])
        pairs.append(dict(verdict="PASS",K=k,seed=seed,pair=str(path),hashes=hashes(path),summary=summary,
            inputs=dict(reference=by_key[k,113]["hashes"],candidate=by_key[k,seed]["hashes"]),
            summary_input_reference=by_key[k,113]["run"],summary_input_candidate=by_key[k,seed]["run"]))
    return runs,pairs


def test_full_matrix_counts_and_undefined_denominators(tmp_path):
    runs,pairs=fixture_summary_inputs(tmp_path,MATRIX)
    result=aggregate(tmp_path/"summary",runs,pairs)
    assert result["status"]=="COMPLETE"
    for row in result["descriptive_statistics"]:
        assert row["observed_pairs"]==4
        if row["metric"]=="lesion_pearson":
            assert row["defined_values"]==3 and row["undefined_values"]==1
    assert {r["K"] for r in result["descriptive_statistics"]}=={5,7}


def test_changed_pair_never_declares_complete(tmp_path):
    runs,pairs=fixture_summary_inputs(tmp_path,[(5,113),(5,211)])
    (Path(pairs[0]["pair"])/"tampered.txt").write_text("changed")
    with pytest.raises(ValueError,match="changed pair"):aggregate(tmp_path/"summary",runs,pairs)


def test_launch_arguments_for_every_seed(queue,monkeypatch):
    w,_=queue
    # Restore the actual launcher, but stub only the native execution boundary.
    calls=[]
    monkeypatch.setattr(w,"invoke",lambda command,label,seed,log:calls.append((command,seed,log)))
    for k,s in MATRIX:Workflow.launch(w,k,s)
    for (k,s),(command,process_seed,log) in zip(MATRIX,calls):
        assert command[command.index("--k")+1]==str(k)
        assert command[command.index("--seed")+1]==str(s)==str(process_seed)
        assert command[command.index("--output-dir")+1]==str(w.run_path(k,s))
        assert log.parent==w.state/"training_logs"


def test_clean_worktree_guard_is_not_bypassed(tmp_path,monkeypatch):
    w=Workflow(tmp_path/"repo",tmp_path/"state","a"*40)
    def git(*args):
        if args[0]=="branch":return "journal-research-local"
        if args[0]=="rev-parse":return "a"*40
        return " M research/code/runner.py"
    monkeypatch.setattr(w,"git",git)
    with pytest.raises(ValueError,match="Dirty source"):w.assert_source()


def test_missing_pair_with_all_runs_is_incomplete(tmp_path):
    runs,pairs=fixture_summary_inputs(tmp_path,MATRIX)
    result=aggregate(tmp_path/"summary",runs,pairs[:-1])
    assert result["status"]=="INCOMPLETE"
    assert all(r["observed_pairs"]==3 for r in result["descriptive_statistics"] if r["K"]==7)
