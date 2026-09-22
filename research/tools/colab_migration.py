"""Supervised Colab readiness and manually enabled remaining queue."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.colab_runtime import ColabQueue
from research.code.workflow_evidence import write_new

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['readiness','launch'])
    for name in ('payload','workspace','durable','dataset'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--readiness',type=Path)
    p.add_argument('--reviewed-readiness-sha256')
    p.add_argument('--single-runtime-confirmation')
    p.add_argument('--result-output', type=Path, help='New JSON result file, separate from progress and diagnostics')
    a=p.parse_args();q=ColabQueue(a.payload,a.workspace,a.durable,a.dataset)
    if a.result_output and a.result_output.exists():
        p.error('Result output exists; review before repeating')
    if a.action=='readiness': result=q.readiness()
    else:
        if not all((a.readiness,a.reviewed_readiness_sha256,a.single_runtime_confirmation)):
            p.error('Launch requires readiness path, reviewed hash and explicit single-runtime confirmation')
        result=q.run(a.readiness,a.reviewed_readiness_sha256,a.single_runtime_confirmation)
    if a.result_output:
        write_new(a.result_output, result)
    print(json.dumps(result,indent=2), flush=True)
