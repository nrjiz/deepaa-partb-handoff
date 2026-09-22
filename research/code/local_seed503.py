"""D-021: one manually supervised Windows seed503 attempt, never a queue."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .workflow import Workflow, WorkflowLock, competing, resources, PACKAGES, stamp
from .workflow_evidence import require, read, write_new, sha, hashes, validate_run
from .colab_transfer import preserve, verify_tree, copy_new
from .colab_runtime import verify_images

CONFIRMATION = 'K5 seed503 belongs to Windows only; no Colab or other project training is running.'
DATASET_SHA256 = '33e276582dd05053ccbd0914d227086188e63013ea1aa16dad5b6e56430c4647'
ORIGINAL = {
    '.RUNNING': 'df52a72cd9136b632ee96a6438b2a4ed132fc1997f209f930608cf6252725c41',
    'configuration.json': '80453a626e704cf3d6924d8d5851343be5dcbcdaedce4f5f0f209cf8c2d00625',
    'progress.jsonl': '95ac2406382acb40e9122696b98cf858f16cb235b20429f9b1c9fb8a3e65eaa9',
    'training_history.csv': '402f0ddb25310f13f38af6ae2628d5a5cea691c531304d16fe089ccedbaf2197',
}


def snapshot():
    require(os.name == 'nt', 'Windows only')
    script = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,Name,CommandLine) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',script],
                            capture_output=True,text=True)
    require(result.returncode == 0, 'Process identity query failed: '+result.stderr)
    rows = json.loads(result.stdout)
    return rows if isinstance(rows,list) else [rows]


def no_competitors(rows):
    require(not competing(rows), 'Competing process: '+json.dumps(competing(rows)))
    lookup = {int(r['ProcessId']):r for r in rows}
    own = set(); pid = os.getpid()
    while pid in lookup and pid not in own:
        own.add(pid); pid = int(lookup[pid]['ParentProcessId'])
    for r in rows:
        if int(r['ProcessId']) in own:
            continue
        command = r.get('CommandLine') or ''
        if r['Name'].lower() in ('powershell.exe','pwsh.exe','cmd.exe'):
            require(bool(command), 'Cannot inspect shell command line; launch blocked')
            require(not re.search(r'local_seed503\.py|run_stability|colab_migration\.py|stability_workflow\.py.*finish',command,re.I),
                    'Competing launcher: '+json.dumps(r))


class LocalSeed503:
    def __init__(self, root, state, head):
        self.w = Workflow(root,state,head)
        self.folder = self.w.state/'local_seed503'
        self.original = self.w.run_root/'K5/seed_503'

    def predecessor(self):
        require(hashes(self.original) == ORIGINAL, 'Original seed503 evidence changed')
        markers = list(self.w.run_root.rglob('.RUNNING'))
        require(markers == [self.original/'.RUNNING'], 'Other active/interrupted markers block launch')
        require(not list((self.w.run_root/'K5').glob('seed_503_*')), 'Existing sibling blocks another attempt')

    def inspect(self, inventory):
        require(sha(inventory)==DATASET_SHA256, 'Dataset inventory differs from reviewed transfer')
        self.predecessor()
        rows = snapshot(); no_competitors(rows)
        require(Path(sys.executable).resolve()==self.w.python.resolve(), 'Use reproduction interpreter')
        require(sys.version_info[:3]==(3,13,3), 'Python version changed')
        versions = {n:importlib.metadata.version(n) for n in PACKAGES}
        require(versions == PACKAGES, 'Package versions changed')
        r = resources(self.w.root)
        require(r['available_ram_bytes']>=1024**3, 'Less than 1 GiB RAM available')
        require(r['disk_free_bytes']>=2*1024**3, 'Less than 2 GiB free disk')
        from .data import load_journal_v1
        split = self.w.root/'research/splits/journal_v1'
        frames = load_journal_v1(split/'train.csv',split/'validation.csv',split/'test.csv',self.w.image_dirs)
        spec = read(inventory)
        from .constants import MANIFEST_SHA256
        require(spec['manifests']==MANIFEST_SHA256 and len(spec['files'])==10015, 'Dataset inventory scope')
        verify_images(self.w.data,spec)
        subprocess.run([str(self.w.python),'-B','-m','pip','check'],check=True)
        from .recovery import registered_path
        registered_path(self.w.root,self.w.state,5,401,self.w.run_root/'K5/seed_401')
        lock = self.w.state/'workflow.lock.json'
        record = read(lock) if lock.exists() else None
        return dict(utc=stamp(),head=self.w.git('rev-parse','HEAD'),
            source_status=self.w.git('status','--porcelain=v1','--untracked-files=all'),
            resources=r,versions=versions,python=sys.version,executable=sys.executable,
            partitions={n:len(f) for n,f in frames.items()},dataset_inventory_sha256=sha(inventory),
            processes=rows,lock=record,lock_owner_identity_live=bool(record and any(
                int(p['ProcessId'])==record['pid'] and p['CreationDate']==record['creation'] for p in rows)),
            original_hashes=ORIGINAL,primary_training='NOT RUN')

    def guard(self, inventory):
        result = self.inspect(inventory)
        self.w.assert_source()
        require(not (self.w.state/'workflow.lock.json').exists(), 'Workflow lock present; senior guarded release required')
        return result

    def register(self, inventory):
        require(not self.folder.exists(), 'Registration/preservation already exists; no second attempt')
        result = self.inspect(inventory)
        self.w.assert_source()
        require(not result['lock_owner_identity_live'], 'Original lock owner is live')
        require(sha(self.w.state/'workflow.lock.json') ==
                '57c9edccbe282cd8531697be220250d43e39a5a00c9eaf7560fc4ca6c6dc2de4',
                'Original lock differs; preserve and review before registration')
        # Exclusive folder creation is the permanent one-attempt reservation.
        # Snapshot the ORIGINAL lock before its separate senior-only guarded release.
        self.folder.mkdir(exist_ok=False)
        preserve(self.w.root,self.w.state,self.folder/'preserved')
        copy_new(inventory,self.folder/'DATASET.json')
        write_new(self.folder/'environment.json',result)
        attempt = 'seed_503_attempt_'+stamp()
        record = dict(schema='local_seed503_v1',decision='D-021',K=5,latent_dimension=4,seed=503,
            owner='Windows',start_epoch=1,model_resume=False,head=self.w.head,
            root=str(self.w.root),original=str(self.original),original_hashes=ORIGINAL,
            original_status='INTERRUPTED_NO_RESUMABLE_CHECKPOINT',
            output=str(self.original.with_name(attempt)),log=str(self.folder/(attempt+'.log')),
            preservation_sha256=sha(self.folder/'preserved/PRESERVATION.json'),
            dataset_inventory_sha256=sha(self.folder/'DATASET.json'),
            environment_sha256=sha(self.folder/'environment.json'),created_utc=stamp())
        write_new(self.folder/'REGISTRATION.json',record)
        return record

    def registration(self):
        r = read(self.folder/'REGISTRATION.json')
        require((r['schema'],r['decision'],r['K'],r['latent_dimension'],r['seed'],r['owner'],r['start_epoch'],r['model_resume']) ==
                ('local_seed503_v1','D-021',5,4,503,'Windows',1,False), 'Registration identity/scope changed')
        require(r['head']==self.w.head and r['root']==str(self.w.root), 'Registration source changed')
        require(r['original']==str(self.original) and r['original_hashes']==ORIGINAL and
                r['original_status']=='INTERRUPTED_NO_RESUMABLE_CHECKPOINT', 'Predecessor binding changed')
        output = Path(r['output'])
        require(output.parent==self.original.parent and re.fullmatch(r'seed_503_attempt_\d{8}T\d{12}Z',output.name), 'Invalid sibling path')
        require(Path(r['log'])==self.folder/(output.name+'.log'), 'Log mapping changed')
        for name,key in [('DATASET.json','dataset_inventory_sha256'),('environment.json','environment_sha256'),
                         ('preserved/PRESERVATION.json','preservation_sha256')]:
            require(sha(self.folder/name)==r[key], 'Registration evidence changed')
        preserved = read(self.folder/'preserved/PRESERVATION.json')
        for item in preserved['files']:
            # The original workflow lock was explicitly released; its preserved copy remains bound.
            require(sha(self.folder/'preserved'/item['copy'])==item['sha256'], 'Preservation copy changed')
            if item['copy'].split('/')[0] not in ('lock',):
                require(sha(Path(item['source']))==item['sha256'], 'Original evidence changed')
        for name, expected in preserved['original_directory_inventories'].items():
            verify_tree(Path(name), expected)
        return r

    def launch(self, confirmation):
        require(confirmation==CONFIRMATION, 'Explicit Windows-only ownership confirmation required')
        r = self.registration()
        require(not (self.folder/'LAUNCH.json').exists(), 'Attempt already claimed; no repeat or partial retry')
        require(not Path(r['output']).exists() and not Path(r['log']).exists(), 'Output/log exists')
        result = self.guard(self.folder/'DATASET.json')
        rows=snapshot(); no_competitors(rows)
        lock=WorkflowLock(self.w.state,self.w.head,rows); lock.acquire()
        child=None
        try:
            write_new(self.folder/'LAUNCH.json',dict(utc=stamp(),registration_sha256=sha(self.folder/'REGISTRATION.json'),
                confirmation=confirmation,environment=result,execution='Windows CPU',start_epoch=1))
            command=self.w.command('run_stability_one.py','--k',5,'--seed',503,'--output-dir',r['output'],
                                   '--image-dirs',*self.w.image_dirs)
            env=dict(os.environ,PYTHONHASHSEED='503',TF_DETERMINISTIC_OPS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1')
            print('Single Windows K5/seed503 attempt. Output: '+r['output']+'; log: '+r['log'],flush=True)
            with Path(r['log']).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=self.w.root,env=env,stdout=log,stderr=subprocess.STDOUT)
                write_new(self.folder/'PROCESS.json',dict(pid=child.pid,command=command,utc=stamp(),
                    identities=[p for p in snapshot() if int(p['ProcessId'])==child.pid]))
                code=child.wait()
            write_new(self.folder/'EXIT.json',dict(exit_code=code,utc=stamp()))
            require(code==0,'Training child failed; preserve evidence; no retry')
            self.w.assert_source()
            require(hashes(self.original)==ORIGINAL,'Original changed during training')
            # Unchanged numerical validator, fresh CPU-only interpreter; zero training steps.
            command=self.w.command('local_seed503.py','validate','--approved-head',self.w.head)
            print('Training child exited successfully; running unchanged artifact validator.',flush=True)
            with (self.folder/'validation.log').open('x') as log:
                child=subprocess.Popen(command,cwd=self.w.root,env=env,stdout=log,stderr=subprocess.STDOUT)
                code=child.wait()
            require(code==0,'Post-run validation failed; no retry')
            return read(self.folder/'VALIDATION.json')
        finally:
            # Interrupted supervisor cannot release while its child might still run.
            if child is None or child.poll() is not None:
                lock.release()

    def validate(self):
        r=self.registration(); self.w.assert_source()
        require((self.folder/'LAUNCH.json').is_file() and read(self.folder/'EXIT.json')['exit_code']==0,'No successful launch exit')
        require(read(self.folder/'LAUNCH.json')['registration_sha256']==sha(self.folder/'REGISTRATION.json'),
                'Registration changed after launch')
        result=validate_run(self.w.root,Path(r['output']),5,503,[self.w.head])
        require(result['verdict']=='PASS_WITH_LIMITATIONS','Validation failed')
        write_new(self.folder/'VALIDATION.json',result)
        return result
