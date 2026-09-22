"""D-020 Linux orchestration. Drive records are NOT cross-runtime locks.

Primary execution requires a supervised single-runtime declaration. No takeover,
automatic retry, partial-run resume, Windows mutation or scientific overrides.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .colab_transfer import inventory, verify_tree, copy_new, stamp, git, verify_index
from .workflow_evidence import read, write_new, require, sha, hashes

CONFIRMATION = 'I deleted or disconnected all previous project runtimes and will use only this project session.'
QUEUE = [(5,503),(7,113),(7,211),(7,307),(7,401),(7,503)]
STUDY = 'deepaa-stability-v1-D020'


@contextlib.contextmanager
def local_lock(path=Path('/tmp/deepaa-stability-v1-D020.lock')):
    require(sys.platform.startswith('linux'), 'Colab execution requires Linux')
    import fcntl
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another local project operation owns the lock') from exc
        yield


def process_snapshot():
    result=[]
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            raw=(p/'stat').read_text(); fields=raw[raw.rfind(')')+2:].split()
            command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            result.append(dict(pid=int(p.name),ppid=int(fields[1]),start_ticks=int(fields[19]),
                               boot_id=boot,executable=os.readlink(p/'exe'),command=command))
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError(f'Cannot inspect process identity {p}; no launch') from exc
    return result


def no_competing():
    rows=process_snapshot(); lookup={p['pid']:p for p in rows}; own=set(); pid=os.getpid()
    while pid in lookup and pid not in own:
        own.add(pid);pid=lookup[pid]['ppid']
    competing=[r for r in rows if r['pid'] not in own and any(t in r['command'] for t in
               ('colab_gpu_worker.py','colab_migration.py','run_stability_one.py','run_stability_matrix.py','recover_seed401_exports.py'))]
    require(not competing, f'Competing project process: {competing}')
    return rows


def source_guard(root, package):
    require(package['release'] and not package['source_status'], 'Draft package cannot launch')
    require(git(root,'rev-parse','HEAD')==package['head'],'Reviewed HEAD differs')
    verify_index(root, package['head'])
    require(not git(root,'status','--porcelain=v1','--untracked-files=all'),'Dirty source/index')
    from .constants import BASELINE_SOURCE, BASELINE_SOURCE_SHA256, MANIFEST_SHA256
    require(sha(root/BASELINE_SOURCE)==BASELINE_SOURCE_SHA256,'Protected source bytes')
    for n,h in MANIFEST_SHA256.items():
        require(sha(root/'research/splits/journal_v1'/n)==h,'Frozen manifest changed')
    for name,entry in package['files'].items():
        if name.startswith('source_bytes/'):
            require(sha(root/name.removeprefix('source_bytes/'))==entry['sha256'],'Reviewed source snapshot changed')


def verify_images(dataset, spec):
    for rel,record in spec['files'].items():
        path=dataset/rel
        require(path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],f'Dataset bytes differ: {rel}')
    meta=dataset/'HAM10000_metadata.csv'
    require(meta.stat().st_size==spec['metadata']['bytes'] and sha(meta)==spec['metadata']['sha256'],'Metadata bytes differ')
    for partition in ('HAM10000_images_part_1','HAM10000_images_part_2'):
        require({f'{partition}/{p.name}' for p in (dataset/partition).glob('*.jpg')}==
                {n for n in spec['files'] if n.startswith(partition+'/')},'Unexpected/missing JPEG names')


def prepare_images(drive_dataset, local_dataset, spec):
    print('Dataset: verifying Drive source images and metadata.', flush=True)
    verify_images(drive_dataset,spec)
    require(shutil.disk_usage(local_dataset.parent).free > sum(v['bytes'] for v in spec['files'].values())+4*1024**3,
            'Insufficient runtime-local data/output headroom')
    local_dataset.mkdir(exist_ok=False)
    for index, rel in enumerate([*spec['files'],'HAM10000_metadata.csv'], 1):
        copy_new(drive_dataset/rel,local_dataset/rel)
        if index % 1000 == 0:
            print(f'Dataset: copied and hashed {index} files.', flush=True)
    print('Dataset: verifying runtime-local inventory.', flush=True)
    verify_images(local_dataset,spec)


def save_snapshot(source, destination):
    """Immutable content-addressed copies. Partial snapshots never imply completion."""
    destination.mkdir(parents=True,exist_ok=True)
    objects=destination/'objects'; objects.mkdir(exist_ok=True)
    entries={}
    if not source.exists():
        return entries
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name.endswith('.tmp'):
            continue
        require(not path.is_symlink(),'Snapshot symlink')
        before=path.stat()
        data=path.read_bytes()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            continue  # A later poll records the finished write; never call this final.
        digest=hashlib.sha256(data).hexdigest(); target=objects/digest
        if not target.exists():
            with target.open('xb') as handle:
                handle.write(data);handle.flush();os.fsync(handle.fileno())
        require(sha(target)==digest,'Durable object readback failed')
        entries[path.relative_to(source).as_posix()]=dict(bytes=len(data),sha256=digest)
    snapshots=destination/'snapshots';snapshots.mkdir(exist_ok=True)
    write_new(snapshots/(stamp()+'.json'),dict(utc=stamp(),files=entries,status='PARTIAL_SNAPSHOT_NOT_RESUMABLE'))
    return entries


def restore_objects(source, entries, target):
    require(not target.exists(),'Refusing to merge durable outputs')
    target.mkdir(parents=True)
    for name,item in entries.items():
        dest=(target/name).resolve()
        require(dest.is_relative_to(target.resolve()),'Unsafe durable path')
        copy_new(source/'objects'/item['sha256'],dest)
    verify_tree(target,entries)


def verify_durable(source, receipt):
    require(receipt.get('status','VERIFIED_DURABLE_COMPLETE')=='VERIFIED_DURABLE_COMPLETE','Not durable completion')
    for name,item in receipt['files'].items():
        p=source/'objects'/item['sha256']
        require(p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],f'Durable hash mismatch: {name}')


def verify_registration(record,k,s,package_hash,head,predecessor):
    require((record['K'],record['latent_dimension'],record['seed'])==(k,k-1,s),'Registration identity')
    require(record['primary_identity']==f'K{k}_seed{s}' and record['decision']=='D-020','Registration scope')
    require(record['package_sha256']==package_hash and record['head']==head,'Registration provenance')
    require(record['start_epoch']==1 and record['model_resume'] is False,'No model resume')
    require(re.fullmatch(rf'seed_{s}_attempt_\d{{8}}T\d{{12}}Z_[0-9a-f]{{8}}',record['attempt_name']), 'Unsafe attempt name')
    if (k,s)==(5,503):
        require(record['predecessor']==dict(path='research/runs/stability_v1/K5/seed_503',
                classification='INTERRUPTED_NO_RESUMABLE_CHECKPOINT',hashes=predecessor),'Predecessor binding')
    else:
        require(record['predecessor'] is None,'Unexpected predecessor')


def worker(root, workspace, mode, arguments, log, *, seed=113, monitor=None):
    env=dict(os.environ,PYTHONHASHSEED=str(seed),TF_DETERMINISTIC_OPS='1',PYTHONDONTWRITEBYTECODE='1')
    # Historical artifact checks use CPU inference, never CPU training.
    if mode in ('validate','pair_verify','aggregate'):
        env['CUDA_VISIBLE_DEVICES']='-1'
    command=[sys.executable,'-B','-u',str(root/'research/tools/colab_gpu_worker.py'),mode,*map(str,arguments)]
    print(f'Worker starting: {mode}; log={log}', file=sys.stderr, flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as handle:
        child=subprocess.Popen(command,cwd=workspace/'aliases',env=env,stdout=handle,stderr=subprocess.STDOUT)
        meta=dict(command=command,pid=child.pid,started_utc=stamp(),process=None)
        # Fast exits can be observed without inventing a creation identity.
        for row in process_snapshot():
            if row['pid']==child.pid:
                meta['process']=row
        write_new(log.with_suffix('.process.json'),meta)
        errors=[]
        while child.poll() is None:
            if monitor:
                try: monitor()
                except Exception as exc:
                    errors.append(repr(exc))
            time.sleep(2)
        code=child.wait()
    if monitor:
        try: monitor()
        except Exception as exc: errors.append(repr(exc))
    write_new(log.with_suffix('.exit.json'),dict(exit_code=code,utc=stamp(),persistence_errors=errors))
    print(f'Worker finished: {mode}; exit={code}; log={log}', file=sys.stderr, flush=True)
    if code or errors:
        with log.open('rb') as handle:
            handle.seek(max(0, log.stat().st_size - 12000))
            print(handle.read().decode(errors='replace'), file=sys.stderr, flush=True)
    require(code==0 and not errors,f'Worker failed or persistence failed; no next run. See {log}; {errors[:1]}')


class ColabQueue:
    def __init__(self,payload,workspace,durable,dataset):
        self.payload=Path(payload).resolve();self.workspace=Path(workspace).resolve()
        self.root=self.workspace/'repo';self.durable=Path(durable).resolve();self.dataset=Path(dataset).resolve()
        self.package=read(self.payload/'PACKAGE.json')
        self.package_hash=sha(self.payload/'PACKAGE.json')
        self.external=self.workspace/'external/stability_workflow_local'
        self.images=[self.dataset/f'HAM10000_images_part_{i}' for i in (1,2)]
        self.runroot=self.root/'research/runs/stability_v1'
        self.session=None

    def guard(self):
        source_guard(self.root,self.package)
        restored=read(self.workspace/'RESTORE.json')
        require(restored['package_sha256']==self.package_hash and
                restored['relocation_sha256']==sha(self.payload/'RELOCATION.json'),'Restoration mapping changed')
        no_competing()
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(mem['MemAvailable'].split()[0])*1024>=1024**3,'Less than 1 GiB available RAM')
        require(shutil.disk_usage(self.workspace).free>3*1024**3,'Insufficient local headroom')
        require(shutil.disk_usage(self.durable.parent).free>3*1024**3,'Insufficient durable headroom')
        original=self.runroot/'K5/seed_503'
        require(hashes(original)==self.package['preserved_seed503'],'Interrupted predecessor changed')
        for marker in self.runroot.rglob('.RUNNING'):
            require(marker==original/'.RUNNING',f'Unresolved local attempt: {marker}')
        known={self.runroot/'K5'/f'seed_{s}' for s in (113,211,307,401,503)}
        for k,s in QUEUE:
            registration=self.durable/'attempts'/f'K{k}_seed{s}'/'REGISTRATION.json'
            if registration.exists():
                record=read(registration)
                verify_registration(record,k,s,self.package_hash,self.package['head'],self.package['preserved_seed503'])
                known.add(self.runroot/f'K{k}'/record['attempt_name'])
        for path in self.runroot.glob('K*/seed_*'):
            require(path in known,f'Unregistered local run blocks queue: {path}')
            if path in (original,self.runroot/'K5/seed_401'):
                continue  # 401 is still gated by its unchanged D-019 receipt verifier.
            require(path.is_dir() and (path/'COMPLETED.json').is_file() and not (path/'FAILED.json').exists(),
                    f'Unregistered/partial local run blocks queue: {path}')

    def call(self,mode,args,label,seed=113,monitor=None):
        worker(self.root,self.workspace,mode,args,self.session/(label+'.log'),seed=seed,monitor=monitor)

    def check_existing_attempts(self):
        """Check the whole queue before launching anything, not only the next seed."""
        expected={f'K{k}_seed{s}':(k,s) for k,s in QUEUE}
        for folder in (self.durable/'attempts').glob('*'):
            require(folder.name in expected and folder.is_dir(),'Unknown durable attempt; review required')
            require((folder/'DURABLE_COMPLETE.json').is_file(),'Existing unresolved attempt blocks launch: '+folder.name)
            k,s=expected[folder.name]
            registration=read(folder/'REGISTRATION.json')
            verify_registration(registration,k,s,self.package_hash,self.package['head'],self.package['preserved_seed503'])
            receipt=read(folder/'DURABLE_COMPLETE.json')
            require(receipt['registration_sha256']==sha(folder/'REGISTRATION.json'),'Registration changed')
            verify_durable(folder,receipt)
            require(sha(Path(receipt['archive']))==receipt['archive_sha256'],'Durable archive changed/missing')

    def old_path(self,k,s):
        if (k,s)==(5,401):
            return self.external/'recovered_exports/K5_seed401_20260915'
        return self.runroot/f'K{k}'/f'seed_{s}'

    def validate(self,k,s,path,label):
        report=self.session/(label+'.json')
        self.call('validate',[path,k,s,self.package['head'],report],label,s)
        r=read(report)
        require(r['verdict']=='PASS_WITH_LIMITATIONS' and r['hashes']==hashes(path),'Invalid/stale validation')
        return r

    def readiness(self):
        with local_lock():
            self.durable.mkdir(parents=True,exist_ok=True)
            self.guard()
            require(not (self.durable/'QUEUE_ACTIVE.json').exists(),'Unresolved durable queue; senior review required')
            self.check_existing_attempts()
            session=self.workspace/'readiness'/stamp();session.mkdir(parents=True)
            self.session=session
            spec=read(self.payload/'DATASET.json');verify_images(self.dataset,spec)
            data_report=session/'data_readiness.json'
            self.call('data',[self.dataset,data_report],'data_readiness')
            env=session/'environment.json'
            self.call('environment',[env],'environment')
            benchmarks={}
            for k,s in ((5,503),(7,113)):
                outputs=[]
                for repeat in (1,2):
                    p=session/f'benchmark_K{k}_{repeat}.json'
                    self.call('benchmark',[k,s,self.dataset,p],f'benchmark_K{k}_{repeat}',s)
                    outputs.append(read(p))
                require(outputs[0]['fingerprint']==outputs[1]['fingerprint'],f'GPU deterministic repeat mismatch K{k}')
                benchmarks[str(k)]=outputs
            # Real existing evidence validation before any primary launch.
            runs=[]
            for s in (113,211,307,401):
                runs.append(self.validate(5,s,self.old_path(5,s),f'old_K5_seed{s}'))
            pairs=[]
            for s in (211,307,401):
                pair=self.root/'research/analysis/stability_v1'/('K5_113_vs_211_20260913' if s==211 else f'K5_113_vs_{s}_workflow_v1')
                report=session/f'old_pair_{s}.json'
                self.call('pair_verify',[pair,self.old_path(5,113),self.old_path(5,s),5,s,report],f'old_pair_{s}')
                pairs.append(read(report))
            record=dict(status='READY_FOR_SENIOR_REVIEW_NOT_LAUNCH',utc=stamp(),package_sha256=self.package_hash,
                        head=self.package['head'],environment=read(env),benchmarks=benchmarks,
                        dataset_inventory_sha256=sha(self.payload/'DATASET.json'),old_runs=runs,old_pairs=pairs,
                        data=read(data_report),
                        runtime_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        process_snapshot=process_snapshot(),single_runtime_confirmation_required=CONFIRMATION,
                        limitations=['Mixed CPU/GPU execution; no CPU/GPU numerical equivalence claim.',
                        'No atomic cross-runtime lock. User must disconnect/delete other project runtimes.',
                        'Workflow continuation is not model resume. Colab lifetime not guaranteed.'])
            write_new(session/'READINESS.json',record)
            save_snapshot(session,self.durable/'readiness'/session.name)
            return dict(path=str(session/'READINESS.json'),sha256=sha(session/'READINESS.json'),**record)

    def run(self,readiness,reviewed_sha,confirmation):
        require(confirmation==CONFIRMATION,'Explicit single-runtime declaration missing')
        require(sha(readiness)==reviewed_sha,'Readiness hash has not been explicitly reviewed')
        ready=read(readiness)
        require(ready['package_sha256']==self.package_hash and ready['head']==self.package['head'],'Readiness/package mismatch')
        require(ready['runtime_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'Readiness from another runtime')
        with local_lock():
            self.guard()
            require(not (self.durable/'QUEUE_ACTIVE.json').exists(),'Unresolved queue record; no automatic takeover')
            self.check_existing_attempts()
            # Exclusive create is ONLY local bookkeeping; cross-runtime safety is manual.
            session_id=stamp()+'_'+uuid.uuid4().hex[:8]
            self.session=self.workspace/'sessions'/session_id;self.session.mkdir(parents=True)
            active=dict(session=session_id,head=self.package['head'],package_sha256=self.package_hash,
                        readiness_sha256=reviewed_sha,confirmation=confirmation,utc=stamp(),processes=process_snapshot())
            write_new(self.durable/'QUEUE_ACTIVE.json',active)
            write_new(self.session/'invocation.json',active)
            try:
                verify_images(self.dataset,read(self.payload/'DATASET.json'))
                environment=self.session/'environment.json'
                self.call('environment',[environment],'environment')
                require(read(environment)['freeze']==ready['environment']['freeze'],'Environment changed after readiness')
                runs={}
                for s in (113,211,307,401):
                    runs[5,s]=self.validate(5,s,self.old_path(5,s),f'validate_old_{s}')
                for k,s in QUEUE:
                    self.guard()
                    identity=f'K{k}_seed{s}'; durable=self.durable/'attempts'/identity
                    if durable.exists():
                        require((durable/'DURABLE_COMPLETE.json').is_file(),'Existing unresolved attempt blocks launch: '+identity)
                        receipt=read(durable/'DURABLE_COMPLETE.json');verify_durable(durable,receipt)
                        require(sha(Path(receipt['archive']))==receipt['archive_sha256'],'Durable archive changed/missing')
                        registration=read(durable/'REGISTRATION.json')
                        verify_registration(registration,k,s,self.package_hash,self.package['head'],self.package['preserved_seed503'])
                        require(receipt['registration_sha256']==sha(durable/'REGISTRATION.json'),'Registration changed')
                        require(registration['package_sha256']==self.package_hash,'Continuation requires original reviewed package')
                        path=self.runroot/f'K{k}'/registration['attempt_name']
                        if path.exists(): verify_tree(path,receipt['files'])
                        else: restore_objects(durable,receipt['files'],path)
                        runs[k,s]=self.validate(k,s,path,f'validate_durable_{identity}')
                        continue
                    durable.mkdir(parents=True,exist_ok=False)
                    attempt_name=f'seed_{s}_attempt_{session_id}'
                    path=self.runroot/f'K{k}'/attempt_name
                    require(not path.exists(),'Attempt output already exists')
                    registration=dict(K=k,latent_dimension=k-1,seed=s,attempt_name=attempt_name,
                        primary_identity=identity,package_sha256=self.package_hash,head=self.package['head'],decision='D-020',
                        start_epoch=1,model_resume=False,session=session_id,log=str(self.session/(identity+'.log')),
                        predecessor=dict(path='research/runs/stability_v1/K5/seed_503',classification='INTERRUPTED_NO_RESUMABLE_CHECKPOINT',
                                         hashes=self.package['preserved_seed503']) if (k,s)==(5,503) else None)
                    write_new(durable/'REGISTRATION.json',registration)
                    verify_registration(registration,k,s,self.package_hash,self.package['head'],self.package['preserved_seed503'])
                    require(read(durable/'REGISTRATION.json')==registration,'Registration readback')
                    def persist():
                        save_snapshot(path,durable)
                        save_snapshot(self.session,self.durable/'sessions'/session_id)
                    self.call('train',[k,s,path,self.dataset],identity,s,monitor=persist)
                    self.guard()
                    report=self.validate(k,s,path,f'validate_{identity}')
                    entries=save_snapshot(path,durable)
                    require(entries==inventory(path),'Final durable copy incomplete')
                    receipt=dict(status='VERIFIED_DURABLE_COMPLETE',utc=stamp(),files=entries,
                        validation=report,registration_sha256=sha(durable/'REGISTRATION.json'))
                    verify_durable(durable,receipt)
                    # Archive before allowing the next primary run.
                    from .workflow_evidence import archive_unit
                    archive=archive_unit(self.durable/'archives',identity,dict(run=path,session=self.session),entries)
                    receipt.update(archive=str(archive),archive_sha256=sha(archive))
                    write_new(durable/'DURABLE_COMPLETE.json',receipt)
                    require(read(durable/'DURABLE_COMPLETE.json')==receipt,'Completion receipt readback')
                    runs[k,s]=report
                pairs=[]
                for k in (5,7):
                    for s in (211,307,401,503):
                        ref=Path(runs[k,113]['run']);cand=Path(runs[k,s]['run'])
                        if k==5 and s in (211,307,401):
                            pair=self.root/'research/analysis/stability_v1'/('K5_113_vs_211_20260913' if s==211 else f'K5_113_vs_{s}_workflow_v1')
                        else:
                            pair=self.root/'research/analysis/stability_v1'/f'K{k}_113_vs_{s}_D020_{session_id}'
                            self.call('compare',[ref,cand,pair],f'compare_K{k}_{s}')
                        result=self.session/f'pair_K{k}_{s}.json'
                        self.call('pair_verify',[pair,ref,cand,k,s,result],f'pair_K{k}_{s}')
                        r=read(result);r.update(summary_input_reference=str(ref),summary_input_candidate=str(cand));pairs.append(r)
                        from .workflow_evidence import archive_unit
                        archive_unit(self.durable/'archives',f'pair_K{k}_{s}_{session_id}',dict(pair=pair,verification=result),r['hashes'])
                inputs=self.session/'summary_inputs.json';write_new(inputs,dict(runs=list(runs.values()),pairs=pairs))
                summary=self.workspace/'summaries'/session_id
                self.call('aggregate',[inputs,summary],'aggregate')
                # Add audit outside the sealed numerical summary, never rewrite it.
                bundle=self.durable/'summaries'/session_id;bundle.mkdir(parents=True,exist_ok=False)
                for p in summary.rglob('*'):
                    if p.is_file(): copy_new(p,bundle/'metrics'/p.relative_to(summary))
                write_new(bundle/'ATTEMPT_AUDIT.json',dict(decision='D-020',predecessor=self.package['preserved_seed503'],
                    attempts=[read(self.durable/'attempts'/f'K{k}_seed{s}'/'REGISTRATION.json') for k,s in QUEUE],
                    limitation='Mixed CPU/GPU environments confound seed-only interpretation; no numerical-equivalence claim.'))
                write_new(bundle/'BUNDLE_SHA256.json',hashes(bundle))
                save_snapshot(self.session,self.durable/'sessions'/session_id)
                # Normal successful closure only. Any failure retains QUEUE_ACTIVE.
                closed=self.durable/'closed_queues';closed.mkdir(exist_ok=True)
                require(read(self.durable/'QUEUE_ACTIVE.json')==active,'Queue ownership record changed')
                (self.durable/'QUEUE_ACTIVE.json').rename(closed/(session_id+'.json'))
                return dict(status='COMPLETE',bundle=str(bundle))
            except BaseException as exc:
                write_new(self.session/'STOPPED.json',dict(utc=stamp(),error=repr(exc),note='No automatic retry, lock takeover or model resume'))
                save_snapshot(self.session,self.durable/'sessions'/session_id)
                raise
