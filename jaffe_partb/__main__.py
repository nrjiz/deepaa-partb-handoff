"""Run `python -m jaffe_partb plan` first. No command launches a seed queue."""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
import hashlib
from .common import ROOT, config, external, inventory, read, require, sha, verify, write


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    default=str(ROOT/'jaffe_partb/proposal.json')
    q=sub.add_parser('plan');q.add_argument('--config',default=default)
    q=sub.add_parser('audit-data');q.add_argument('--config',default=default);q.add_argument('--data',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('train');q.add_argument('--config',default=default);q.add_argument('--manifest',required=True);q.add_argument('--data',required=True);q.add_argument('--approval',required=True);q.add_argument('--workspace',required=True);q.add_argument('--k',type=int,required=True);q.add_argument('--seed',type=int,required=True)
    q=sub.add_parser('validate');q.add_argument('--config',default=default);q.add_argument('--run',required=True);q.add_argument('--data',required=True);q.add_argument('--receipt',required=True)
    q=sub.add_parser('pair')
    for name in ('reference','candidate','reference-receipt','candidate-receipt','output'):q.add_argument('--'+name,required=True)
    q=sub.add_parser('verify-pair');q.add_argument('--pair',required=True);q.add_argument('--receipt',required=True)
    q=sub.add_parser('aggregate');q.add_argument('--pair',nargs='+',required=True);q.add_argument('--receipt',nargs='+',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('figures');q.add_argument('--pair',required=True);q.add_argument('--receipt',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('preserve');q.add_argument('--source',required=True);q.add_argument('--archive',required=True)
    args=p.parse_args()
    for name in ('output','workspace','receipt','archive'):
        value=getattr(args,name,None)
        if value and not isinstance(value,list):external(value)
    if args.command=='plan':
        from .data import parse_name,coverage
        c=config(args.config)
        if c['dataset']=='JAFFE':
            meta=read(ROOT/'jaffe_partb/official_filename_inventory.json');counts=coverage([parse_name(n) for n in meta['filenames']],c)
        else:
            from research.code.constants import EXPECTED_PARTITION_COUNTS,MANIFEST_SHA256
            counts=dict(images=EXPECTED_PARTITION_COUNTS,frozen_split_hashes=MANIFEST_SHA256)
        print(json.dumps(dict(status='PROPOSED_NOT_AUTHORIZED',config=c,config_sha256=sha(args.config),metadata_coverage=counts,
            intended_runs=[dict(K=k,latent_dimension=k-1,seed=s,output=f"<workspace>/{c['experiment_id']}/runs/K{k}_seed{s}") for k in c['K'] for s in c['seeds']],
            real_images_audited=False,training_started=False),indent=2));return
    if args.command=='audit-data':
        from .data import audit
        print(json.dumps(audit(args.data,config(args.config),sha(args.config),args.output)['coverage'],indent=2));return
    if args.command=='train':
        from .training import train
        train(args);return
    if args.command=='validate':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Fresh CPU-only validation process required')
        from .training import validate_run
        write(args.receipt,validate_run(args.run,args.data,args.config));return
    if args.command=='preserve':
        verify(args.source);before=inventory(Path(args.source))
        require(not Path(args.archive).resolve().is_relative_to(Path(args.source).resolve()),'Archive must be outside source')
        with zipfile.ZipFile(args.archive,'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(args.source).rglob('*')):
                if f.is_file():z.write(f,f.relative_to(args.source).as_posix())
        with zipfile.ZipFile(args.archive) as z:
            require(z.testzip() is None,'Archive CRC')
            for n,item in before.items():require(hashlib.sha256(z.read(n)).hexdigest()==item['sha256'],'Archive readback')
        require(before==inventory(Path(args.source)),'Source changed during preservation')
        write(str(args.archive)+'.receipt.json',dict(status='HASH_READBACK_PASS',sha256=sha(args.archive),independent_backup='NOT_ESTABLISHED_BY_THIS_COMMAND',source_inventory_sha256=sha(Path(args.source)/'INVENTORY.json')));return
    from . import analysis
    if args.command=='verify-pair':write(args.receipt,analysis.verify_pair(args.pair))
    else:getattr(analysis,args.command)(args)


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
