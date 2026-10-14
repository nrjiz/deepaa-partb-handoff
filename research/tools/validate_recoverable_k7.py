"""Fresh CPU-only final artifact validation for explicitly labeled Route B outputs."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.code.workflow_evidence import require,read,write_new
if __name__=='__main__':
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Validation must be CPU-only')
    from research.code.recoverable_validation import validate
    run,auth,identity,destination=map(Path,sys.argv[1:])
    a=read(auth);i=read(identity)
    write_new(destination,validate(ROOT,run,[i['head']],a,i))
