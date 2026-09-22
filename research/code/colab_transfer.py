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
import sys
import struct
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


def run_visible(command, *, cwd=None, env=None, log=None):
    """Stream merged subprocess output to the notebook and an optional new log."""
    import contextlib
    print('Running: ' + ' '.join(map(str, command)), flush=True)
    with (Path(log).open('x', encoding='utf-8') if log else contextlib.nullcontext()) as handle:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as child:
            for line in child.stdout:
                print(line, end='', flush=True)
                if handle:
                    handle.write(line); handle.flush()
            code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, command, output=f'Full output: {log}' if log else None)


def verify_index(root, head):
    """Compare every index mode/blob/stage/path with the reviewed commit, read-only."""
    require(git(root, 'rev-parse', 'HEAD') == head, 'Restored HEAD differs')
    index = subprocess.check_output(['git', '--no-optional-locks', 'ls-files', '--stage', '-z'], cwd=root)
    tree = subprocess.check_output(['git', '--no-optional-locks', 'ls-tree', '-r', '-z', head], cwd=root)
    expected = {}
    for entry in tree.split(b'\0'):
        if entry:
            meta, path = entry.split(b'\t', 1)
            mode, kind, digest = meta.split()
            expected[path] = (mode, digest, b'0')
    actual = {}
    for entry in index.split(b'\0'):
        if entry:
            meta, path = entry.split(b'\t', 1)
            require(path not in actual, 'Unmerged index')
            actual[path] = tuple(meta.split())
    require(actual == expected, 'Index entries differ from reviewed commit')
    return hashlib.sha256(index).hexdigest()


def snapshot_attributes(source):
    """Describe exact snapshot EOLs before checkout; never rewrite snapshot bytes."""
    rules = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b'\0' in data or b'\n' not in data:
            continue
        crlf = data.count(b'\r\n')
        name = path.relative_to(source).as_posix()
        rules.append(json.dumps(name) + (' text eol=crlf\n' if crlf else ' text eol=lf\n'))
    return ''.join(rules)


def refresh_snapshot_stat_cache(root, source, head):
    """Invalidate only stat sizes, then let Git refresh, after content identity proof.

    Git treats a nonzero cached size mismatch as modified before comparing filtered
    content. Mixed EOL snapshots cannot always have checkout's size. No blob ID,
    mode, stage, flag, path or working-tree byte is changed by this operation.
    """
    before = verify_index(root, head)
    changed = subprocess.check_output(['git','--no-optional-locks','diff-files','--name-only','-z'],cwd=root)
    names = [n.decode('utf-8') for n in changed.split(b'\0') if n]
    if not names:
        return
    for name in names:
        require((source/name).is_file() and sha(root/name)==sha(source/name), 'Changed file outside exact source snapshot')
        require(git(root,'hash-object','--path='+name,name)==git(root,'rev-parse',head+':'+name),
                'Snapshot filtered content differs from reviewed commit')
    path = root/'.git/index'
    original = path.read_bytes()
    require(original[:4]==b'DIRC' and struct.unpack('!I',original[4:8])[0]==2,
            'Expected disposable index format v2')
    require(hashlib.sha1(original[:-20]).digest()==original[-20:], 'Index checksum mismatch')
    data = bytearray(original); offset = 12
    for _ in range(struct.unpack('!I',data[8:12])[0]):
        start = offset
        flags = struct.unpack('!H',data[start+60:start+62])[0]
        require(not flags & 0x4000, 'Extended index entry requires review')
        end = data.index(0,start+62)
        name = bytes(data[start+62:end]).decode('utf-8')
        if name in names:
            data[start+36:start+40] = b'\0'*4  # stat size only; NOT the blob or flags
        offset = start + ((end+1-start+7)//8)*8
    data[-20:] = hashlib.sha1(data[:-20]).digest()
    # Use Git's lock convention; never overwrite a competing index lock.
    lock = root/'.git/index.lock'
    with lock.open('xb') as handle:
        require(path.read_bytes()==original, 'Index changed concurrently')
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(lock,path)
    subprocess.run(['git','update-index','--refresh'],cwd=root,check=True)
    require(verify_index(root,head)==before, 'Stat refresh changed index entries')
    for name in names:
        require(sha(root/name)==sha(source/name), 'Stat refresh changed snapshot bytes')


def checkout_source(payload, workspace):
    """Disposable checkout, also testable on Windows; original repositories are untouched."""
    m = read(payload/'PACKAGE.json')
    require(m['release'] and not m['source_status'], 'Draft package cannot execute')
    actual=inventory(payload); actual.pop('PACKAGE.json')
    require(actual==m['files'], 'Package content changed')
    workspace.mkdir(parents=True,exist_ok=False)
    root = workspace/'repo'
    print('Restore: payload verified; checking out reviewed bundle.', flush=True)
    subprocess.run(['git','clone','--no-checkout',str(payload/'history.bundle'),str(root)],check=True)
    subprocess.run(['git','config','core.autocrlf','input'],cwd=root,check=True)
    subprocess.run(['git','config','core.eol','lf'],cwd=root,check=True)
    subprocess.run(['git','config','index.version','2'],cwd=root,check=True)
    subprocess.run(['git','config','core.splitIndex','false'],cwd=root,check=True)
    # Select each file's checkout EOL from the verified snapshot, including the
    # packaged notebook. A release may contain both LF and CRLF source files.
    with (root/'.git/info/attributes').open('x', encoding='utf-8') as handle:
        handle.write(snapshot_attributes(payload/'source_bytes'))
    subprocess.run(['git','checkout','-b','colab-reviewed',m['head']],cwd=root,check=True)
    subprocess.run(['git','remote','remove','origin'],cwd=root,check=True)
    index_before = verify_index(root, m['head'])
    for p in (payload/'source_bytes').rglob('*'):
        if p.is_file():
            # Exact reviewed working-tree bytes, never normalization of the payload.
            dest=root/p.relative_to(payload/'source_bytes')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(p.read_bytes())
    require(verify_index(root, m['head']) == index_before, 'Overlay changed index entries')
    refresh_snapshot_stat_cache(root, payload/'source_bytes', m['head'])
    require(not git(root,'status','--porcelain=v1','--untracked-files=all'), 'Restored source not Git-clean')
    for name, entry in m['files'].items():
        if name.startswith('source_bytes/'):
            dest = root/name.removeprefix('source_bytes/')
            require(dest.stat().st_size == entry['bytes'] and sha(dest) == entry['sha256'], 'Restored source snapshot differs')
    require(sha(root/BASELINE_SOURCE)==BASELINE_SOURCE_SHA256,'Protected CRLF source hash')
    for n,h in MANIFEST_SHA256.items():
        require(sha(root/'research/splits/journal_v1'/n)==h,'Manifest bytes')
    print('Restore: clean index/HEAD and all exact source/manifest hashes verified.', flush=True)
    return root


def restore(payload, workspace):
    """Linux-only exact byte restoration; no scientific source/hash substitution."""
    require(sys.platform.startswith('linux'), 'Restore is Linux-only')
    root = checkout_source(payload, workspace)
    m = read(payload/'PACKAGE.json')
    print('Restore: copying historical evidence and resolving exact path aliases.', flush=True)
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
    # Run unchanged D-019 registration/receipt checks in the historical alias CWD,
    # using restored code in a fresh interpreter, before environment installation.
    code = '''
import sys
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0,str(root))
from research.code.recovery import registered_path
from research.code.workflow_evidence import read, sha, require
state=root.parent/'external/stability_workflow_local'
actual=registered_path(root,state,5,401,root/'research/runs/stability_v1/K5/seed_401')
require(actual==(state/'recovered_exports/K5_seed401_20260915').resolve(),'D-019 registration missing')
entry=read(state/'recoveries/K5_seed401.json')
require(sha(Path(entry['archive']))==entry['archive_sha256'],'D-019 archive changed')
print('D-019 registration, receipt, original/external evidence and archive verified; no model loaded.')
'''
    run_visible([sys.executable, '-B', '-u', '-c', code, str(root)], cwd=aliases,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), log=workspace/'restore_d019.log')
    verify_index(root, m['head'])
    require(not git(root,'status','--porcelain=v1','--untracked-files=all'), 'Restored evidence dirtied source')
    write_new(workspace/'RESTORE.json',dict(head=m['head'],package_sha256=sha(payload/'PACKAGE.json'),
               relocation_sha256=sha(payload/'RELOCATION.json'),
               index_entries_sha256=verify_index(root, m['head']), d019_registration_receipt='VERIFIED',
               byte_policy='exact source snapshot; per-file checkout EOL from verified bytes'))
    print('Restore complete. GPU readiness and primary launch remain pending.', flush=True)
    return root
