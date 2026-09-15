#!/usr/bin/env python3
"""Local frozen-matrix workflow. Only 'finish' can launch training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.workflow import Workflow, processes, release_stale, classify, MATRIX
from research.code.workflow_evidence import require, write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["status","watch","preflight","finish","summarize","update-docs","release-stale-lock","_validate","_verify_pair"])
    parser.add_argument("--approved-head",help="Explicit reviewed full Git revision; mandatory except status")
    parser.add_argument("--state-dir",type=Path,default=ROOT.parent/"stability_workflow_local")
    parser.add_argument("--summary-dir",type=Path)
    parser.add_argument("--k",type=int,choices=[5,7])
    parser.add_argument("--seed",type=int,choices=[113,211,307,401,503])
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    w=Workflow(ROOT,args.state_dir,args.approved_head)
    if args.action=="status":
        print(json.dumps(w.status(),indent=2));return
    if args.action=="watch":w.watch();return
    require(args.approved_head is not None,"--approved-head is required")
    if args.action in {"_validate","_verify_pair"}:
        # Read-only numerical workers; never invoke primary training or alter source.
        require(args.k is not None and args.seed is not None and args.report is not None,"Worker needs k/seed/report")
        require(args.report.resolve().is_relative_to(w.state),"Worker reports must be in external state")
        from research.code.workflow_evidence import validate_run,verify_pair
        if args.action=="_validate":result=validate_run(ROOT,w.run_path(args.k,args.seed),args.k,args.seed,w.allowed_heads(args.k,args.seed))
        else:result=verify_pair(ROOT,w.pair_path(args.k,args.seed),w.run_path(args.k,113),w.run_path(args.k,args.seed),args.k,args.seed)
        write_new(args.report,result);print(json.dumps(dict(verdict=result["verdict"],report=str(args.report))));return
    if args.action=="preflight":
        resource=w.guard(check_data=True)
        require(not (w.state/"workflow.lock.json").exists(),"Workflow lock exists; inspect status")
        for k,seed in MATRIX:require(classify(w.run_path(k,seed)) in {"MISSING","COMPLETED_UNVERIFIED","RECOVERED_UNVERIFIED"},"Partial/active run blocks queue")
        for k,seed in MATRIX:
            if seed==113:continue
            pair=w.pair_path(k,seed)
            if pair.exists():
                for name in ["alignment/permutation.json","alignment/cost_matrix.csv","alignment/cost_method_matrix.csv",
                             "alignment/candidate_validation_coefficients_aligned.npy","alignment/candidate_test_coefficients_aligned.npy",
                             "test/stability_summary.json","test/coefficient_stability_by_lesion.csv",
                             "test/semantic_stability.csv","test/nearest_real_stability.csv","test/visual_stability.csv"]:
                    require((pair/name).is_file(),f"Partial comparison, preserve and investigate: {pair/name}")
        print(json.dumps(dict(status="READY_FOR_VALIDATION_AND_QUEUE",resources=resource,
            note="Completed runs/pairs still require full validation before skip; no training performed",inventory=w.status()),indent=2));return
    if args.action=="release-stale-lock":
        w.assert_source();release_stale(w.state,processes());print("Stale lock preserved under released_locks; no process stopped.");return
    if args.action=="update-docs":
        require(args.summary_dir is not None,"--summary-dir required")
        print(json.dumps(w.update_docs(args.summary_dir),indent=2));return
    w.execute(args.action)


if __name__=="__main__":
    try:main()
    except KeyboardInterrupt:
        print("INTERRUPTED: inspect child processes and partial outputs; do not assume resume or cleared lock.",file=sys.stderr);sys.exit(130)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}",file=sys.stderr);sys.exit(1)
