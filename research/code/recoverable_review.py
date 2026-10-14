"""Restore draft bytes for review only; never forge a clean committed checkout."""
from pathlib import Path
import shutil
from .colab_transfer import extract_verified,inventory
from .workflow_evidence import read,require,sha,write_new

def restore_review(release,destination):
    release,destination=Path(release),Path(destination)
    receipt=read(release/'TRANSFER_RECEIPT.json')
    require(receipt['release'] is False,'This review helper is only for disabled drafts')
    require(sha(release/'stability_v1_colab.ipynb')==receipt['notebook_sha256'],'Notebook mismatch')
    destination.mkdir(parents=True,exist_ok=False)
    payload=destination/'payload';m=extract_verified(release/receipt['archive'],receipt['sha256'],payload)
    require(sha(payload/'PACKAGE.json')==receipt['package_manifest_sha256'],'Manifest mismatch')
    require(m['head']==receipt['head'] and m['release'] is False and bool(m['source_status']),'Draft provenance differs')
    shutil.copytree(payload/'source_bytes',destination/'review_source')
    require(inventory(payload/'source_bytes')==inventory(destination/'review_source'),'Review byte restoration differs')
    write_new(destination/'REVIEW_RESTORE.json',dict(status='REVIEW_BYTES_ONLY_NOT_EXECUTABLE',base_head=m['head'],source_status=m['source_status'],package_sha256=receipt['package_manifest_sha256'],source_inventory=inventory(destination/'review_source')))
    return destination/'review_source'
