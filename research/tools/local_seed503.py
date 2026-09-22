"""D-021 single Windows seed503 attempt. Only explicit 'launch' trains."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.local_seed503 import LocalSeed503

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','preflight','register','launch','validate'])
    p.add_argument('--approved-head',required=True)
    p.add_argument('--dataset-inventory',type=Path)
    p.add_argument('--confirmation')
    a=p.parse_args()
    task=LocalSeed503(ROOT,ROOT.parent/'stability_workflow_local',a.approved_head)
    if a.action in ('inspect','preflight','register'):
        if not a.dataset_inventory:p.error('--dataset-inventory required')
        result=getattr(task,{'preflight':'guard'}.get(a.action,a.action))(a.dataset_inventory)
    elif a.action=='launch':result=task.launch(a.confirmation)
    else:result=task.validate()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
