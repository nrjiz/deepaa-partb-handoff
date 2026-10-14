"""Explicit reviewed Route B child process only; drafts always rejected."""
import argparse,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.code.recoverable_workflow import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('payload','workspace','dataset','durable','authorization'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reviewed-authorization-sha256',required=True)
    p.add_argument('--k',type=int,choices=[7],required=True)
    p.add_argument('--latent-dimension',type=int,choices=[6],required=True)
    p.add_argument('--seed',type=int,choices=[113],required=True)
    a=p.parse_args();run(ROOT,a.payload,a.workspace,a.dataset,a.durable,a.authorization,a.reviewed_authorization_sha256)
