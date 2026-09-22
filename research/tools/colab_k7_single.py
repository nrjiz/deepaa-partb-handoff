"""Explicit K7/latent6/seed113 readiness or one manual launch; no queue."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.colab_k7_single import K7Single, IDENTITY
from research.code.workflow_evidence import write_new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['readiness','launch'])
    for name in ('payload','workspace','durable','dataset'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--k',type=int,choices=[7],required=True)
    p.add_argument('--latent-dimension',type=int,choices=[6],required=True)
    p.add_argument('--seed',type=int,choices=[113],required=True)
    p.add_argument('--readiness',type=Path)
    p.add_argument('--reviewed-readiness-sha256')
    p.add_argument('--single-runtime-confirmation')
    p.add_argument('--result-output',type=Path,required=True)
    a=p.parse_args()
    if a.result_output.exists():p.error('Result output exists; preserve and review before repeating')
    q=K7Single(a.payload,a.workspace,a.durable,a.dataset)
    if a.action=='readiness':result=q.readiness()
    else:
        if not all((a.readiness,a.reviewed_readiness_sha256,a.single_runtime_confirmation)):
            p.error('Launch needs reviewed readiness and explicit single-runtime confirmation')
        result=q.run(a.readiness,a.reviewed_readiness_sha256,a.single_runtime_confirmation)
    write_new(a.result_output,result)
    print(json.dumps(result,indent=2),flush=True)
