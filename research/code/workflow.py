"""Sequential local workflow; source stays pinned and clean until explicit docs update."""
from __future__ import annotations

import ctypes
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .workflow_evidence import require, read, write_new, sha, hashes, archive_unit, aggregate

MATRIX = tuple((k,s) for k in (5,7) for s in (113,211,307,401,503))
HISTORICAL = {
    (5,113): "0e32c8d469be141d30e627219aeac7ee253a34fa",
    (5,211): "f097f2f9e6ad72464685dacca59ba95ba273ab33",
    (5,307): "1a7214d89ac2d5d52f6161b19f86db71ed144ab7",
}
PACKAGES = {"tensorflow":"2.21.0", "keras":"3.15.1", "numpy":"2.5.2",
            "pandas":"3.0.5", "scipy":"1.18.1", "h5py":"3.14.0", "pillow":"12.3.0"}
LEGACY_LOGS = {113:"primary_K5_seed113_0e32c8d_20260905_a7f19c.log",
               211:"primary_K5_seed211_20260909_b6e42f.log",307:"primary_K5_seed307_20260913_c8d241.log"}


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def processes():
    require(os.name == "nt", "This local workflow requires Windows process identity checks")
    script = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CreationDate,Name,CommandLine) | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",script],capture_output=True,text=True,check=True)
    value=json.loads(result.stdout)
    return value if isinstance(value,list) else [value]


def competing(rows):
    """Exclude only our process and ancestors (including Windows venv launcher)."""
    lookup={int(r["ProcessId"]):r for r in rows}
    ancestors=set();pid=os.getpid()
    while pid in lookup and pid not in ancestors:
        ancestors.add(pid);pid=int(lookup[pid]["ParentProcessId"])
    viewer_script=str(Path(__file__).resolve().parents[1]/"tools/stability_workflow.py").replace("\\","/").lower()
    def viewer(row):
        command=(row.get("CommandLine") or "").replace("\\","/").lower()
        return bool(re.search(r'"?'+re.escape(viewer_script)+r'"?\s+watch(?:\s|$)',command))
    return [r for r in rows if int(r["ProcessId"]) not in ancestors and not viewer(r) and
            (re.match(r"^(python(?:w|\d+(?:\.\d+)?)?|wsl)\.exe$",r["Name"],re.I) or
             (r["Name"].lower() not in {"powershell.exe","pwsh.exe"} and
              re.search(r"run_stability|train\.py|benchmark\.py",r.get("CommandLine") or "",re.I)))]


def resources(root):
    class Memory(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ["total","available","page_total","page_available","virtual_total","virtual_available","extended"]]
    m=Memory();m.length=ctypes.sizeof(m)
    require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))),"Cannot inspect RAM")
    return dict(available_ram_bytes=m.available,total_ram_bytes=m.total,memory_load_percent=m.load,
                disk_free_bytes=shutil.disk_usage(root).free)


def classify(path):
    if not path.exists():return "MISSING"
    if (path/".RUNNING").exists():return "ACTIVE_OR_INTERRUPTED"
    if (path/"FAILED.json").exists():return "FAILED_PARTIAL"
    if (path/"COMPLETED.json").exists():return "COMPLETED_UNVERIFIED"
    return "PARTIAL"  # Includes empty pre-created directories; never silently reused.


class WorkflowLock:
    """Exclusive-create lock; abandoned locks require explicit, identity-checked release."""
    def __init__(self,state,head,rows):
        self.state=state;self.path=state/"workflow.lock.json"
        own=next((r for r in rows if int(r["ProcessId"])==os.getpid()),None)
        require(own is not None,"Cannot establish workflow process identity")
        self.record=dict(pid=os.getpid(),creation=own["CreationDate"],head=head,created_utc=stamp())

    def acquire(self):
        self.state.mkdir(parents=True,exist_ok=True)
        write_new(self.path,self.record)

    def release(self):
        require(read(self.path)==self.record,"Lock ownership changed")
        folder=self.state/"released_locks";folder.mkdir(exist_ok=True)
        self.path.rename(folder/(stamp()+".json"))


def release_stale(state,rows):
    path=state/"workflow.lock.json";record=read(path)
    # PID reuse is not sufficient to declare a lock live; match creation identity too.
    live=any(int(r["ProcessId"])==record["pid"] and r["CreationDate"]==record["creation"] for r in rows)
    require(not live,"Lock owner is still alive; refusing release")
    require(not competing(rows),"Other Python/training process active; refusing stale-lock release")
    destination=state/"released_locks";destination.mkdir(exist_ok=True)
    path.rename(destination/("stale_"+stamp()+".json"))


class Workflow:
    def __init__(self,root,state,head):
        self.root=Path(root).resolve();self.state=Path(state).resolve();self.head=head
        self.python=self.root.parent/".venvs/deepaa_data2026_repro/Scripts/python.exe"
        self.data=Path("C:/Users/UserPc/Downloads/archive")
        self.image_dirs=[self.data/"HAM10000_images_part_1",self.data/"HAM10000_images_part_2"]
        self.run_root=self.root/"research/runs/stability_v1"
        self.analysis=self.root/"research/analysis/stability_v1"
        self.session=None;self.child=None
        require(not self.state.is_relative_to(self.root),"Workflow state must be outside source repository")

    def run_path(self,k,seed):return self.run_root/f"K{k}"/f"seed_{seed}"

    def pair_path(self,k,seed):
        if (k,seed)==(5,211):return self.analysis/"K5_113_vs_211_20260913"
        return self.analysis/f"K{k}_113_vs_{seed}_workflow_v1"

    def git(self,*args):
        return subprocess.check_output(["git",*args],cwd=self.root,text=True).strip()

    def assert_source(self):
        require(bool(re.fullmatch(r"[0-9a-f]{40}",self.head or "")),"An explicitly approved full HEAD is required")
        require(self.git("branch","--show-current")=="journal-research-local","Wrong branch")
        require(self.git("rev-parse","HEAD")==self.head,"HEAD differs from approved pin")
        # Git stderr is not discarded: the known ignored pytest cache warning remains visible.
        require(not self.git("status","--porcelain=v1","--untracked-files=all"),"Dirty source/index: review and commit manually")
        from .constants import BASELINE_SOURCE,BASELINE_SOURCE_SHA256,MANIFEST_SHA256
        require(sha(self.root/BASELINE_SOURCE)==BASELINE_SOURCE_SHA256,"Protected source hash")
        for name,digest in MANIFEST_SHA256.items():
            require(sha(self.root/"research/splits/journal_v1"/name)==digest,"Manifest hash")

    def guard(self,check_data=False):
        rows=processes()
        require(not competing(rows),"Active Python/training process: "+json.dumps(competing(rows)))
        markers=list(self.run_root.rglob(".RUNNING"))
        require(not markers,"Active/interrupted run markers; preserve and investigate: "+str(markers))
        self.assert_source()
        require(Path(sys.executable).resolve()==self.python.resolve(),"Use the documented reproduction interpreter")
        require(sys.version_info[:3]==(3,13,3),"Python version differs from audited environment")
        for name,version in PACKAGES.items():
            require(importlib.metadata.version(name)==version,f"Environment changed: {name}")
        r=resources(self.root)
        # Operational safety floors, not scientific hyperparameters or peak-memory promises.
        require(r["available_ram_bytes"]>=1024**3,"Less than 1 GiB available RAM; stop and free competing workload manually")
        require(r["disk_free_bytes"]>=2*1024**3,"Less than 2 GiB free disk; preserve evidence and provision space")
        if check_data:
            from .data import load_journal_v1
            split=self.root/"research/splits/journal_v1"
            load_journal_v1(split/"train.csv",split/"validation.csv",split/"test.csv",self.image_dirs)
        return r

    def status(self):
        runs=[dict(K=k,seed=s,state=classify(self.run_path(k,s)),path=str(self.run_path(k,s))) for k,s in MATRIX]
        pairs=[dict(K=k,seed=s,state=("PRESENT_UNVERIFIED" if self.pair_path(k,s).exists() else "MISSING"),path=str(self.pair_path(k,s))) for k,s in MATRIX if s!=113]
        return dict(head=self.git("rev-parse","HEAD"),git_status=self.git("status","--porcelain=v1","--untracked-files=all"),
                    runs=runs,pairs=pairs,lock=read(self.state/"workflow.lock.json") if (self.state/"workflow.lock.json").exists() else None,
                    note="Read-only inventory, not validation; COMPLETED markers alone are not verified completion.")

    def watch(self):
        """Follow the newest workflow session and switch automatically between training logs."""
        import time
        offsets={}
        print("Read-only viewer; Ctrl+C stops this viewer only.",flush=True)
        try:
            while True:
                candidates=[]
                sessions=sorted((self.state/"sessions").glob("*/events.jsonl"))
                training=sorted((self.state/"training_logs").glob("*.log"),key=lambda p:p.stat().st_mtime_ns)
                if sessions:candidates.append(sessions[-1])
                if training:candidates.append(training[-1])
                for path in candidates:
                    with path.open("rb") as handle:
                        if path not in offsets:
                            lines=handle.read().splitlines()[-20:]
                            print(f"--- {path} ---",flush=True)
                            for line in lines:print(line.decode("utf-8",errors="replace"),flush=True)
                        else:
                            handle.seek(offsets[path])
                            chunk=handle.read()
                            if chunk:print(chunk.decode("utf-8",errors="replace"),end="",flush=True)
                        offsets[path]=handle.tell()
                time.sleep(2)
        except KeyboardInterrupt:
            print("Viewer stopped; workflow/training unaffected.",flush=True)

    def emit(self,event,**fields):
        record=dict(utc=stamp(),event=event,**fields)
        print(json.dumps(record),flush=True)
        if self.session:
            with (self.session/"events.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(record)+"\n");f.flush();os.fsync(f.fileno())

    def allowed_heads(self,k,seed):
        heads={self.head}
        if (k,seed) in HISTORICAL:heads.add(HISTORICAL[k,seed])
        for path in (self.state/"approved_pins").glob("*.json"):
            heads.add(read(path)["head"])
        return sorted(heads)

    def command(self,tool,*args):return [str(self.python),"-B","-u",str(self.root/"research/tools"/tool),*map(str,args)]

    def invoke(self,command,label,seed=113,log_path=None):
        """Fresh process, direct file redirection, actual native return code. No pipes."""
        log_path=log_path or self.session/(label+".log")
        env=dict(os.environ,PYTHONHASHSEED=str(seed),TF_DETERMINISTIC_OPS="1",PYTHONDONTWRITEBYTECODE="1")
        self.emit("subprocess_start",label=label,command=command,log=str(log_path),seed=seed)
        with log_path.open("x",encoding="utf-8") as log:
            self.child=subprocess.Popen(command,cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_new(self.session/(label+"_process.json"),dict(pid=self.child.pid,command=command,log=str(log_path)))
            code=self.child.wait()
        self.child=None
        self.emit("subprocess_exit",label=label,exit_code=code)
        require(code==0,f"{label} failed with native exit {code}; inspect {log_path}; no restart/overwrite")

    def validate(self,k,seed):
        self.guard()
        report=self.session/f"validate_K{k}_seed{seed}.json"
        self.invoke(self.command("stability_workflow.py","_validate", "--k",k,"--seed",seed,
                    "--report",report,"--approved-head",self.head,"--state-dir",self.state),f"validate_K{k}_seed{seed}",seed)
        result=read(report)
        require(result["verdict"]=="PASS_WITH_LIMITATIONS" and result["hashes"]==hashes(self.run_path(k,seed)),"Validation receipt failed/stale")
        result["report"]=str(report)
        self.emit("run_verified",K=k,seed=seed,revision=result["execution_revision"])
        return result

    def launch(self,k,seed):
        self.guard(check_data=True)
        require(classify(self.run_path(k,seed))=="MISSING","Run is no longer missing")
        log_dir=self.state/"training_logs";log_dir.mkdir(exist_ok=True)
        log=log_dir/f"K{k}_seed{seed}_{stamp()}.log"
        self.invoke(self.command("run_stability_one.py","--k",k,"--seed",seed,"--output-dir",self.run_path(k,seed),"--image-dirs",*self.image_dirs),f"train_K{k}_seed{seed}",seed,log)

    def training_log(self,k,seed):
        if (k,seed) in HISTORICAL and read(self.run_path(k,seed)/"configuration.json")["implementation_git_commit"]==HISTORICAL[k,seed]:
            path=self.root.parent/LEGACY_LOGS[seed]
            require(path.is_file(),f"Missing original training log {path}")
            return path
        paths=list((self.state/"training_logs").glob(f"K{k}_seed{seed}_*.log"))
        require(len(paths)==1,f"Missing/ambiguous training log for K{k}/seed{seed}: {paths}")
        return paths[0]

    def backup_run(self,report):
        k,s=report["K"],report["seed"]
        sources=dict(run=self.run_path(k,s),training_log=self.training_log(k,s),validation=Path(report["report"]))
        if (k,s)==(5,113):sources["prior_validation"]=self.root.parent/"validation_K5_seed113_20260909"
        if (k,s)==(5,211):sources["prior_validation"]=self.root.parent/"stability_pair_20260913"
        path=archive_unit(self.state/"archives",f"run_K{k}_seed{s}",sources,report["hashes"])
        self.emit("archive_verified",path=str(path),sha256=sha(path),same_disk=True)

    def pair(self,k,seed,create):
        self.guard()
        pair=self.pair_path(k,seed)
        reference=self.run_path(k,113);candidate=self.run_path(k,seed)
        if not pair.exists():
            if not create:return None
            args=["--reference-run",reference,"--candidate-run",candidate]
            self.invoke(self.command("align_validation_coefficients.py",*args,"--output-dir",pair/"alignment"),f"align_K{k}_seed{seed}")
            self.guard()
            self.invoke(self.command("evaluate_stability_pair.py",*args,"--alignment-dir",pair/"alignment","--output-dir",pair/"test"),f"evaluate_K{k}_seed{seed}")
        # Existing partial/stale analyses fail verification; never filled in silently.
        self.guard()
        report=self.session/f"pair_K{k}_seed{seed}.json"
        self.invoke(self.command("stability_workflow.py","_verify_pair","--k",k,"--seed",seed,
                    "--report",report,"--approved-head",self.head,"--state-dir",self.state),f"verify_pair_K{k}_seed{seed}")
        result=read(report)
        require(result["verdict"]=="PASS" and result["hashes"]==hashes(pair),"Pair verification receipt failed/stale")
        result["summary_input_reference"]=str(reference);result["summary_input_candidate"]=str(candidate)
        # Explicit mask/direction live outside legacy/sealed derivatives.
        methods=__import__('pandas').read_csv(pair/"alignment/cost_method_matrix.csv",header=None).to_numpy()
        evidence=self.session/f"pair_K{k}_seed{seed}_direction.json"
        write_new(evidence,dict(candidate_column_for_reference=result["summary"]["permutation"],
                  application="candidate_raw[:, permutation]",test_used_for_matching=False,
                  fallback_mask=(methods=="fallback_2x_mae").astype(int).tolist()))
        path=archive_unit(self.state/"archives",f"pair_K{k}_seed{seed}",dict(pair=pair,verification=report,direction=evidence),dict(inputs=result["inputs"],derived=result["hashes"]))
        self.emit("pair_verified",K=k,seed=seed,path=str(pair),archive=str(path))
        return result

    def work(self,finish):
        """Validation always precedes skip; no candidate is used before reference validation."""
        run_reports=[];pair_reports=[];verified=set()
        for k,seed in MATRIX:
            state=classify(self.run_path(k,seed))
            self.emit("run_status",K=k,seed=seed,state=state)
            if state=="MISSING":
                if not finish:continue
                self.launch(k,seed)
            elif state!="COMPLETED_UNVERIFIED":
                raise ValueError(f"K{k}/seed{seed}: {state}; preserve attempt, no training resume. Investigate manually.")
            report=self.validate(k,seed)
            run_reports.append(report);verified.add((k,seed))
            self.backup_run(report)
            if seed!=113 and (k,113) in verified:
                pair=self.pair(k,seed,create=finish)
                if pair:pair_reports.append(pair)
            # Durable partial record after every verified unit; outside tracked source.
            write_new(self.session/f"checkpoint_{len(run_reports):02}.json",dict(verified_runs=[(r["K"],r["seed"]) for r in run_reports],verified_pairs=[(p["K"],p["seed"]) for p in pair_reports]))
        self.guard()
        destination=self.analysis/"workflow_summaries"/stamp()
        result=aggregate(destination,run_reports,pair_reports)
        write_new(self.session/"result.json",dict(status=result["status"],summary=str(destination),runs=len(run_reports),pairs=len(pair_reports)))
        archive=archive_unit(self.state/"archives","summary",dict(summary=destination,session=self.session),hashes(destination))
        self.emit("workflow_result",status=result["status"],summary=str(destination),archive=str(archive))
        if finish:require(result["status"]=="COMPLETE","Finish did not validate all ten runs/eight pairs")
        return destination

    def execute(self,action):
        self.guard(check_data=True)
        rows=processes();lock=WorkflowLock(self.state,self.head,rows);lock.acquire()
        self.session=self.state/"sessions"/stamp();self.session.mkdir(parents=True,exist_ok=False)
        pins=self.state/"approved_pins";pins.mkdir(exist_ok=True)
        if not (pins/(self.head+".json")).exists():write_new(pins/(self.head+".json"),dict(head=self.head,explicitly_pinned=True))
        write_new(self.session/"invocation.json",dict(action=action,head=self.head,python=str(self.python),resources=resources(self.root)))
        try:
            return self.work(finish=action=="finish")
        except BaseException as exc:
            self.emit("STOPPED",error=repr(exc),note="No automatic retry, deletion, model resume or process termination")
            raise
        finally:
            # If interrupted while child may still exist, preserve the lock for investigation.
            if self.child is None or self.child.poll() is not None:lock.release()

    def update_docs(self,summary_dir):
        self.guard()
        require(not (self.state/"workflow.lock.json").exists(),"Workflow lock present; no documentation edits")
        summary_dir=summary_dir.resolve()
        require(summary_dir.is_relative_to(self.analysis/"workflow_summaries"),"Not a workflow summary")
        s=read(summary_dir/"summary.json")
        for name,digest in read(summary_dir/"summary_files_sha256.json").items():require(sha(summary_dir/name)==digest,"Summary changed")
        for r in s["verified_runs"]:require(hashes(Path(r["run"]))==r["hashes"],"Run changed after summary")
        for p in s["verified_pairs"]:require(hashes(Path(p["pair"]))==p["hashes"],"Pair changed after summary")
        token=sha(summary_dir/"summary.json")
        text=f'\n\n## Workflow execution update {datetime.now(timezone.utc).date()}\n\nStatus: **{s["status"]}**. Verified {len(s["verified_runs"])}/10 runs and {len(s["verified_pairs"])}/8 comparisons. Separate K5/K7 findings, raw metrics, defined denominators and lesion-bootstrap bounds: `{summary_dir}`. No clinical or overall reproducibility claim follows from execution completion.\n\nSummary SHA-256: `{token}`. Source pin: `{self.head}`. Workflow continuation is not model resume. Originals preserved; archives under `{self.state / "archives"}` are same-disk copies, not disk-failure protection.\n'
        run_lines='\n'.join(f'- K{r["K"]}/seed{r["seed"]}: COMPLETED, validated; execution `{r["execution_revision"]}`; `{r["run"]}`.' for r in s["verified_runs"])
        paths=[self.root/"docs/project_state.md",self.root/"docs/experiments/experiment_registry.md",self.root/"docs/gates/gate_1_stability_infrastructure.md",self.root/"docs/experiments/archetype_stability.md"]
        require(all(token not in p.read_text(encoding="utf-8") for p in paths),"This summary is already documented")
        # Explicit user-invoked final step only, never called from finish/summarize.
        lock=WorkflowLock(self.state,self.head,processes());lock.acquire()
        try:
            self.guard()
            for path in paths:
                with path.open("a",encoding="utf-8",newline="\n") as f:f.write(text+run_lines+'\n')
        finally:lock.release()
        return [str(p) for p in paths]
