"""Non-overwriting transfer, exact-byte restoration and explicit path relocation.

No TensorFlow imports, Git mutations in the source repository, or credentials.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone

from .workflow_evidence import require, read, sha, hashes, write_new
from .constants import BASELINE_SOURCE, BASELINE_SOURCE_SHA256, MANIFEST_SHA256


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=root, text=True).strip()


def regular(path):
    require(not path.is_symlink() and not (hasattr(path, 'is_junction') and path.is_junction()),
            f'Link/junction refused: {path}')


def inventory(root):
    root = Path(root)
    regular(root)
    result = {}
    for path in sorted(root.rglob('*')):
        regular(path)
        if path.is_file():
            result[path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def verify_tree(root, expected):
    require(inventory(root) == expected, f'File set/bytes/hash mismatch: {root}')


def copy_new(source, target):
    regular(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    require(sha(source) == sha(target), f'Copy checksum: {source}')


def preserve(root, state, destination):
    """Snapshot old evidence; the original marker/lock/pin are never changed."""
    sources = {
        'run': root/'research/runs/stability_v1/K5/seed_503',
        'training_log': state/'training_logs/K5_seed503_20260915T181345388069Z.log',
        'launch_session': state/'sessions/20260915T181151848061Z',
        'earlier_session': state/'sessions/20260915T133418789705Z',
        'lock': state/'workflow.lock.json', 'pin': state/'approved-head.txt',
        'approved_pin': state/'approved_pins/4a5daa34f68d9a68b3b024bb87c7c9ff5c85ccc2.json',
    }
    original_sets={str(p):inventory(p) for p in sources.values() if p.is_dir()}
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for label, source in sources.items():
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in paths:
            regular(path)
            if not path.is_file():
                continue
            relative = Path(label)/(path.relative_to(source) if source.is_dir() else path.name)
            before = dict(bytes=path.stat().st_size, sha256=sha(path))
            copy_new(path, destination/relative)
            require(before == dict(bytes=path.stat().st_size, sha256=sha(path)), 'Evidence changed during preservation')
            records.append(dict(source=str(path), copy=relative.as_posix(),
                                modified_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), **before))
    for item in records:
        require(sha(Path(item['source'])) == item['sha256'] == sha(destination/item['copy']), 'Preservation verification')
    for name,expected in original_sets.items():
        verify_tree(Path(name),expected)
    receipt = dict(schema='seed503_preservation_v1', decision='D-020', utc=stamp(),
                   classification='INTERRUPTED_NO_RESUMABLE_CHECKPOINT', files=records,
                   original_run_hashes=hashes(sources['run']),original_directory_inventories=original_sets,
                   note='Original evidence retained in place; diagnosis accepted 2026-09-21; no resume state.')
    write_new(destination/'PRESERVATION.json', receipt)
    return receipt


def dataset_inventory(root, dataset):
    records = {}
    ids = []
    membership = {}
    for part, digest in MANIFEST_SHA256.items():
        path = root/'research/splits/journal_v1'/part
        require(sha(path) == digest, 'Frozen manifest bytes')
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                rel = f"{row['source_partition']}/{row['image_id']}.jpg"
                require(rel not in records, 'Duplicate image membership')
                p = dataset/rel
                regular(p)
                records[rel] = dict(bytes=p.stat().st_size, sha256=sha(p))
                ids.append(row['image_id'])
                membership[row['image_id']] = (row['lesion_id'], row['dx'])
    require(len(records) == len(set(ids)) == 10015, 'Expected 10015 unique images')
    meta = dataset/'HAM10000_metadata.csv'
    regular(meta)
    with meta.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 10015 and {r['image_id'] for r in rows} == set(ids), 'Metadata image membership')
    require(all(membership[r['image_id']] == (r['lesion_id'],r['dx']) for r in rows), 'Metadata lesion/diagnosis mismatch')
    return dict(schema='ham10000_source_bytes_v1', files=records,
                metadata=dict(bytes=meta.stat().st_size, sha256=sha(meta)), manifests=MANIFEST_SHA256)


def build_package(root, state, dataset, output, *, approved_head=None):
    root, state, dataset, output = map(lambda p: Path(p).resolve(), (root, state, dataset, output))
    require(not output.is_relative_to(root) and not output.is_relative_to(state), 'Package must be separate')
    head = git(root, 'rev-parse', 'HEAD')
    status = git(root, 'status', '--porcelain=v1', '--untracked-files=all')
    require(git(root, 'branch', '--show-current') == 'journal-research-local', 'Wrong branch')
    require(sha(root/BASELINE_SOURCE) == BASELINE_SOURCE_SHA256, 'Protected bytes')
    from .recovery import registered_path
    require(registered_path(root,state,5,401,root/'research/runs/stability_v1/K5/seed_401') ==
            (state/'recovered_exports/K5_seed401_20260915').resolve(), 'D-019 registration/receipt verification')
    if approved_head:
        require(head == approved_head and not status, 'Release package requires reviewed clean HEAD')
    # A full bundle contains source history, not .git/config, hooks or credentials.
    candidates = [root/'research/runs/stability_v1', root/'research/analysis/stability_v1',
                  state/'recovered_exports/K5_seed401_20260915', root/'.git/objects']
    estimate = sum(p.stat().st_size for d in candidates for p in d.rglob('*') if p.is_file())
    require(shutil.disk_usage(output.parent).free > estimate*4 + 3*1024**3, 'Insufficient package headroom')
    output.mkdir(exist_ok=False)
    payload = output/'payload'
    payload.mkdir()
    preservation = preserve(root, state, payload/'preserved_seed503')
    print('Preservation verified; hashing 10015 source JPEGs (no image copy).', flush=True)
    write_new(payload/'DATASET.json', dataset_inventory(root, dataset))
    source_names = git(root, 'ls-files').splitlines()
    # Draft code is explicit; release builds must have all these files committed.
    source_names += [p.relative_to(root).as_posix() for d in ('research/code','research/tools','research/notebooks','research/tests')
                     if (root/d).exists() for p in (root/d).rglob('*') if p.is_file() and p.suffix in ('.py','.ipynb','.ps1')]
    source_names += ['research/COLAB_MIGRATION.md', 'research/requirements_colab.txt']
    source_names = sorted(set(n for n in source_names if n == BASELINE_SOURCE or
                              n.startswith(('research/code/','research/tools/','research/tests/','research/splits/',
                                            'research/configs/','research/protocols/','research/notebooks/','docs/')) or
                              n in ('research/COLAB_MIGRATION.md','research/requirements_colab.txt','research/requirements_stability_v1.txt')))
    for name in source_names:
        copy_new(root/name, payload/'source_bytes'/name)
    subprocess.run(['git','bundle','create',str(payload/'history.bundle'),'--all'],cwd=root,check=True)
    subprocess.run(['git','bundle','verify',str(payload/'history.bundle')],cwd=root,check=True)
    # Only named scientific evidence; never arbitrary home/config/environment files.
    evidence = [root/'research/runs/stability_v1/K5'/f'seed_{s}' for s in (113,211,307,401,503)]
    evidence += [root/'research/analysis/stability_v1'/p for p in
                 ('K5_113_vs_211_20260913','K5_113_vs_307_workflow_v1','K5_113_vs_401_workflow_v1')]
    evidence += [state/'sessions',state/'training_logs',state/'recoveries',state/'approved_pins',
                 state/'recovered_exports/K5_seed401_20260915',state/'approved-head.txt',state/'workflow.lock.json']
    evidence += [root.parent/p for p in ('primary_K5_seed113_0e32c8d_20260905_a7f19c.log',
                 'primary_K5_seed211_20260909_b6e42f.log','primary_K5_seed307_20260913_c8d241.log')]
    # Original registration's archive is hash-bound and must be retained too.
    reg = read(state/'recoveries/K5_seed401.json')
    require(sha(Path(reg['archive'])) == reg['archive_sha256'], 'Seed401 archive changed')
    evidence.append(Path(reg['archive']))
    relocation = {}
    for source in evidence:
        require(source.exists(), f'Missing required evidence {source}')
        rel = source.relative_to(root.parent)
        target = payload/'evidence'/rel
        relocation[str(source)] = ('repo/'+source.relative_to(root).as_posix()
                                   if source.is_relative_to(root) else 'external/'+rel.as_posix())
        for path in sorted(source.rglob('*')) if source.is_dir() else [source]:
            regular(path)
            if path.is_file():
                dest = target/path.relative_to(source) if source.is_dir() else target
                if not dest.exists():
                    copy_new(path,dest)
                relocation[str(path)] = ('repo/'+path.relative_to(root).as_posix()
                                         if path.is_relative_to(root) else 'external/'+path.relative_to(root.parent).as_posix())
    # Strings in pair receipts may use slash spellings; bind both explicit spellings.
    for old, new in list(relocation.items()):
        relocation[old.replace('\\','/')] = new
    write_new(payload/'RELOCATION.json', relocation)
    manifest = dict(schema='colab_transfer_v1', created_utc=stamp(), head=head,
                    branch='journal-research-local', release=bool(approved_head), source_status=status,
                    windows_approved_pin=(state/'approved-head.txt').read_text().strip(),
                    repository_name=root.name, preserved_seed503=preservation['original_run_hashes'],
                    files=inventory(payload), target_gpu_validation='PENDING', decision='D-020')
    write_new(payload/'PACKAGE.json', manifest)
    archive = output/'colab_transfer.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():
                z.write(p,p.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'ZIP CRC failed')
        for name, entry in manifest['files'].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == entry['sha256'], 'ZIP hash mismatch')
    copy_new(root/'research/notebooks/stability_v1_colab.ipynb',output/'stability_v1_colab.ipynb')
    write_new(output/'TRANSFER_RECEIPT.json',dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
               package_manifest_sha256=sha(payload/'PACKAGE.json'),release=bool(approved_head),head=head,
               notebook_sha256=sha(output/'stability_v1_colab.ipynb')))
    # Verify all originals after packaging, not just their preservation copies.
    for e in preservation['files']:
        require(sha(Path(e['source'])) == e['sha256'], 'Original interrupted evidence changed')
    for name,expected in preservation['original_directory_inventories'].items():
        verify_tree(Path(name),expected)
    return dict(output=str(output), **read(output/'TRANSFER_RECEIPT.json'))


def extract_verified(archive, expected_sha, destination):
    require(sha(archive) == expected_sha, 'Package SHA-256 differs from senior receipt')
    require(not destination.exists(), 'Extraction destination exists')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names)==len(set(names)), 'Duplicate ZIP member')
        for name in names:
            require(z.getinfo(name).orig_filename == name, 'Unsafe normalized ZIP member')
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,
                    'Unsafe ZIP member')
            require((z.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink')
        z.extractall(destination)
    m = read(destination/'PACKAGE.json')
    actual = inventory(destination)
    actual.pop('PACKAGE.json')
    require(actual == m['files'], 'Extracted package differs')
    return m


def restore(payload, workspace):
    """Linux-only exact byte checkout; no scientific source/hash substitution."""
    require(os.name == 'posix', 'Restore is Linux-only')
    m = read(payload/'PACKAGE.json')
    require(m['release'] and not m['source_status'], 'Draft package cannot execute')
    actual=inventory(payload); actual.pop('PACKAGE.json')
    require(actual==m['files'], 'Package content changed')
    workspace.mkdir(parents=True,exist_ok=False)
    root = workspace/'repo'
    subprocess.run(['git','clone','--no-checkout',str(payload/'history.bundle'),str(root)],check=True)
    subprocess.run(['git','config','core.autocrlf','true'],cwd=root,check=True)
    subprocess.run(['git','config','core.eol','crlf'],cwd=root,check=True)
    subprocess.run(['git','checkout','-b','colab-reviewed',m['head']],cwd=root,check=True)
    subprocess.run(['git','remote','remove','origin'],cwd=root,check=True)
    for p in (payload/'source_bytes').rglob('*'):
        if p.is_file():
            # These are exact reviewed working-tree bytes, Git-clean under CRLF conversion.
            dest=root/p.relative_to(payload/'source_bytes')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(p.read_bytes())
    require(not git(root,'status','--porcelain=v1','--untracked-files=all'), 'Restored source not Git-clean')
    require(sha(root/BASELINE_SOURCE)==BASELINE_SOURCE_SHA256,'Protected CRLF source hash')
    for n,h in MANIFEST_SHA256.items():
        require(sha(root/'research/splits/journal_v1'/n)==h,'Manifest bytes')
    for p in (payload/'evidence').rglob('*'):
        if p.is_file():
            rel=p.relative_to(payload/'evidence')
            dest=root/Path(*rel.parts[1:]) if rel.parts[0]==m['repository_name'] else workspace/'external'/rel
            copy_new(p,dest)
    # All historical path aliases live OUTSIDE the checkout, in the worker CWD.
    # No receipt changes or patched Path/read/verification functions.
    aliases=workspace/'aliases'; aliases.mkdir()
    mapping=read(payload/'RELOCATION.json')
    for old,new in mapping.items():
        target=(workspace/new).resolve()
        require(target.is_relative_to(workspace.resolve()) and target.exists(),'Relocation target missing/outside workspace')
        alias=aliases/old
        require(not Path(old).is_absolute() and '..' not in Path(old).parts,'Unsafe original path')
        alias.parent.mkdir(parents=True,exist_ok=True)
        if not alias.exists():
            alias.symlink_to(target,target_is_directory=target.is_dir())
        else:
            require(alias.resolve()==target,'Conflicting relocation')
    write_new(workspace/'RESTORE.json',dict(head=m['head'],package_sha256=sha(payload/'PACKAGE.json'),
               relocation_sha256=sha(payload/'RELOCATION.json'),byte_policy='exact source snapshot; core.autocrlf=true'))
    return root
