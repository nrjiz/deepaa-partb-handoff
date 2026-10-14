"""Build clean bounded GPU engineering package. No production execution scope."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.colab_transfer import extract_verified,copy_new,inventory,git,stamp
from research.code.workflow_evidence import require,read,write_new,sha
from research.code.recovery import code_hashes
from research.code.constants import BASELINE_SOURCE,BASELINE_SOURCE_SHA256,MANIFEST_SHA256


def build(base, output, approved_head=None):
    require(approved_head, 'Engineering packaging requires an explicit clean committed HEAD')
    base,output=base.resolve(),output.resolve()
    require(not output.is_relative_to(ROOT) and not output.is_relative_to(base), 'Use a new external directory')
    receipt=read(base/'TRANSFER_RECEIPT.json')
    require(receipt['release'], 'Base must be a reviewed release')
    head=git(ROOT,'rev-parse','HEAD');status=git(ROOT,'status','--porcelain=v1','--untracked-files=all')
    if approved_head:require(head==approved_head and not status,'Reviewed clean commit required')
    output.mkdir(exist_ok=False)
    prior=output/'base_payload'
    m=extract_verified(base/receipt['archive'],receipt['sha256'],prior)
    require(m['head']==receipt['head'] and sha(prior/'PACKAGE.json')==receipt['package_manifest_sha256'], 'Base receipt differs')
    require(m['release'] and not m['source_status'],'Base source was not a release')
    old=read(prior/'evidence/stability_workflow_local/recovered_exports/K5_seed401_20260915/RECOVERED_EXPORT.json')
    require(code_hashes(ROOT)==old['code_hashes'], 'D-019-bound source differs; no re-audit authorized')
    require(sha(ROOT/BASELINE_SOURCE)==BASELINE_SOURCE_SHA256,'Protected source differs')
    for n,h in MANIFEST_SHA256.items():require(sha(ROOT/'research/splits/journal_v1'/n)==h,'Frozen manifest differs')
    payload=output/'payload';payload.mkdir()
    # Frozen evidence is copied exclusively from the verified historical package.
    for folder in ('evidence','preserved_seed503'):
        for p in (prior/folder).rglob('*'):
            if p.is_file():copy_new(p,payload/p.relative_to(prior))
    for name in ('DATASET.json','RELOCATION.json'):copy_new(prior/name,payload/name)
    prior_receipt=payload/'prior_release';prior_receipt.mkdir()
    copy_new(prior/'PACKAGE.json',prior_receipt/'PACKAGE.json')
    copy_new(base/'TRANSFER_RECEIPT.json',prior_receipt/'TRANSFER_RECEIPT.json')
    names=git(ROOT,'ls-files','--cached','--others','--exclude-standard').splitlines()
    names=sorted(set(n for n in names if n==BASELINE_SOURCE or n.startswith(('docs/','research/'))
                     and not n.startswith(('research/runs/','research/analysis/','research/smoke_tests/'))))
    for name in names:copy_new(ROOT/name,payload/'source_bytes'/name)
    subprocess.run(['git','bundle','create',str(payload/'history.bundle'),'--all'],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(payload/'history.bundle')],cwd=ROOT,check=True)
    manifest=dict(m,head=head,release=bool(approved_head),source_status=status,created_utc=stamp(),decision='ST-RECOVERY-05 engineering authorization; scientific amendment pending',
        execution_scope=dict(K=7,latent_dimension=6,seed=113,mode='bounded recovery engineering',owner='Colab'),
        windows_seed503_dependency=False,target_gpu_validation='PENDING',purpose='BOUNDED_RECOVERY_ENGINEERING',primary_training_enabled=False,
        prior_release=dict(head=m['head'],zip_sha256=receipt['sha256'],evidence_as_of=m['created_utc']),
        files=inventory(payload))
    write_new(payload/'PACKAGE.json',manifest)
    archive=output/'colab_transfer.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'ZIP CRC failed')
        for name,item in manifest['files'].items():
            require(hashlib.sha256(z.read(name)).hexdigest()==item['sha256'],'ZIP bytes differ')
    copy_new(ROOT/'research/notebooks/recovery_gpu_acceptance.ipynb',output/'recovery_gpu_acceptance.ipynb')
    result=dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,head=head,release=bool(approved_head),
        package_manifest_sha256=sha(payload/'PACKAGE.json'),notebook_sha256=sha(output/'recovery_gpu_acceptance.ipynb'))
    write_new(output/'TRANSFER_RECEIPT.json',result)
    require(sha(base/receipt['archive'])==receipt['sha256'],'Base ZIP changed')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approved-head')
    a=p.parse_args()
    print(json.dumps(build(a.base_release,a.output,a.approved_head),indent=2))
