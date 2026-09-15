"""One-time authorized seed401 checkpoint-only recovery; never trains."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.recovery import recover

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir",type=Path,default=ROOT.parent/"stability_workflow_local")
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(recover(ROOT,args.state_dir,args.output_dir),indent=2))
