"""Build a draft or explicitly reviewed release package; never stage/commit/train."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.colab_transfer import build_package

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('C:/Users/UserPc/Downloads/archive'))
    p.add_argument('--state',type=Path,default=ROOT.parent/'stability_workflow_local')
    p.add_argument('--approved-head',help='Release only: exact senior-reviewed clean commit; omit for disabled draft')
    a=p.parse_args()
    print(json.dumps(build_package(ROOT,a.state,a.dataset,a.output,approved_head=a.approved_head),indent=2))
