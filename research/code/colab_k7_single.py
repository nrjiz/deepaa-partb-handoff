"""D-021: exactly one independent K7/latent6/seed113 Colab attempt."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from .colab_runtime import (ColabQueue, CONFIRMATION, local_lock, no_competing,
    process_snapshot, source_guard, verify_images, save_snapshot, verify_durable)
from .colab_transfer import inventory, stamp
from .workflow_evidence import require, read, write_new, sha, hashes, archive_unit

IDENTITY = dict(K=7, latent_dimension=6, seed=113)
DURABLE = Path('/content/drive/MyDrive/DeepAA_stability_v1_D020')


class K7Single(ColabQueue):
    """Reuses existing workers/validator, without the old six-run queue or K5 dependency."""

    def identity(self):
        require(self.package.get('execution_scope')==dict(**IDENTITY,mode='one manual run',owner='Colab'),
                'Package does not authorize the independent K7 identity')
        config = read(self.root/'research/configs/archetype_stability_v1.json')
        planned = [r for r in config['planned_runs'] if r['K']==7 and r['seed']==113]
        require(len(planned)==1 and all(planned[0][k]==v for k,v in IDENTITY.items()), 'Frozen K7 identity differs')
        require(config['primary_scope']['latent_dimension_rule']=='K - 1', 'Frozen latent rule differs')

    def integrity(self):
        source_guard(self.root,self.package)
        self.identity()
        restored=read(self.workspace/'RESTORE.json')
        require(restored['package_sha256']==self.package_hash and
                restored['relocation_sha256']==sha(self.payload/'RELOCATION.json'), 'Restoration provenance differs')
        require(sha(self.payload/'DATASET.json')==self.package['files']['DATASET.json']['sha256'], 'Packaged dataset inventory changed')
        require(hashes(self.runroot/'K5/seed_503')==self.package['preserved_seed503'], 'Preserved predecessor differs')
        no_competing()
        require(self.durable==DURABLE.resolve(), 'Use the fixed project Drive destination; no alternate namespace')
        require(os.path.ismount('/content/drive'), 'Drive is not mounted at /content/drive')
        require(shutil.disk_usage(self.workspace).free>3*1024**3, 'Less than 3 GiB runtime headroom')
        require(shutil.disk_usage(self.durable.parent).free>3*1024**3, 'Less than 3 GiB Drive filesystem headroom')
        memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(memory['MemAvailable'].split()[0])*1024>=1024**3, 'Less than 1 GiB available RAM')

    def unclaimed(self):
        for name in ('QUEUE_ACTIVE.json','SINGLE_ACTIVE.json'):
            require(not (self.durable/name).exists(), 'Unresolved project launch record: '+name)
        require(not (self.durable/'attempts/K7_seed113').exists(), 'K7/seed113 already claimed or completed; no retraining')
        require(not list((self.runroot/'K7').glob('seed_113*')), 'Local K7/seed113 output already exists; no retry')
        original=self.runroot/'K5/seed_503/.RUNNING'
        require(all(p==original for p in self.runroot.rglob('.RUNNING')), 'Other active/interrupted runtime output')
        historical={self.runroot/'K5'/f'seed_{seed}' for seed in (113,211,307,401,503)}
        require(all(p in historical for p in self.runroot.glob('K*/seed_*')),
                'Unregistered local output requires review, including failed attempts without a running marker')
        for path in (self.durable/'attempts').glob('*'):
            require(path.is_dir() and (path/'DURABLE_COMPLETE.json').is_file(), 'Unresolved durable attempt: '+str(path))
            receipt=read(path/'DURABLE_COMPLETE.json')
            require(sha(path/'REGISTRATION.json')==receipt['registration_sha256'], 'Existing registration changed')
            verify_durable(path,receipt)
            require(sha(Path(receipt['archive']))==receipt['archive_sha256'], 'Existing archive changed')

    def guard(self):
        self.integrity()
        self.unclaimed()

    def readiness(self):
        with local_lock():
            self.guard()
            self.durable.mkdir(parents=True,exist_ok=True)
            session=stamp()+'_'+uuid.uuid4().hex[:8]
            self.session=self.workspace/'readiness'/session
            self.session.mkdir(parents=True,exist_ok=False)
            probe=self.durable/'readiness'/session
            probe.mkdir(parents=True,exist_ok=False)
            value=dict(identity=IDENTITY,head=self.package['head'],package_sha256=self.package_hash,utc=stamp())
            write_new(probe/'DESTINATION_PROBE.json',value)
            require(read(probe/'DESTINATION_PROBE.json')==value, 'Drive write/readback failed')
            verify_images(self.dataset,read(self.payload/'DATASET.json'))
            self.call('data',[self.dataset,self.session/'data.json'],'data')
            self.call('environment',[self.session/'environment.json'],'environment')
            results=[]
            for repeat in (1,2):
                target=self.session/f'benchmark_{repeat}.json'
                self.call('benchmark',[7,113,self.dataset,target],f'benchmark_{repeat}',113)
                results.append(read(target))
            require(results[0]['fingerprint']==results[1]['fingerprint'], 'K7 deterministic repeat mismatch')
            self.guard()
            record=dict(status='READY_FOR_SENIOR_REVIEW_NOT_LAUNCH',identity=IDENTITY,
                head=self.package['head'],package_sha256=self.package_hash,utc=stamp(),
                dataset_inventory_sha256=sha(self.payload/'DATASET.json'),dataset=str(self.dataset),
                durable=str(self.durable),destination_probe=str(probe/'DESTINATION_PROBE.json'),
                destination_probe_sha256=sha(probe/'DESTINATION_PROBE.json'),
                environment=read(self.session/'environment.json'),benchmarks=results,
                data=read(self.session/'data.json'),processes=process_snapshot(),
                runtime_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                windows_seed503_dependency=False,execution='Colab GPU',
                limitations=['Manual single-runtime discipline; Drive records are not an atomic cross-runtime lock.',
                              'CPU/GPU numerical equivalence is not claimed. No combined analysis in this operation.'])
            ready=self.session/'READINESS.json';write_new(ready,record)
            save_snapshot(self.session,probe)
            return dict(path=str(ready),sha256=sha(ready),**record)

    def run(self,readiness,reviewed_sha,confirmation):
        require(confirmation==CONFIRMATION, 'Explicit single-runtime confirmation required')
        require(sha(readiness)==reviewed_sha, 'Readiness hash not reviewed')
        ready=read(readiness)
        require(ready['status']=='READY_FOR_SENIOR_REVIEW_NOT_LAUNCH' and ready['identity']==IDENTITY, 'Readiness identity/status differs')
        require(ready['head']==self.package['head'] and ready['package_sha256']==self.package_hash, 'Readiness source differs')
        require(ready['runtime_boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'Readiness from another runtime')
        require(ready['dataset_inventory_sha256']==sha(self.payload/'DATASET.json') and
                ready['dataset']==str(self.dataset) and ready['durable']==str(self.durable), 'Readiness paths/data differ')
        require(sha(Path(ready['destination_probe']))==ready['destination_probe_sha256'], 'Drive probe changed')
        with local_lock():
            self.guard()
            verify_images(self.dataset,read(self.payload/'DATASET.json'))
            session=stamp()+'_'+uuid.uuid4().hex[:8]
            self.session=self.workspace/'sessions'/session;self.session.mkdir(parents=True,exist_ok=False)
            environment=self.session/'environment.json'
            self.call('environment',[environment],'environment')
            require(read(environment)['freeze']==ready['environment']['freeze'], 'Environment changed since readiness')
            self.guard()
            durable=self.durable/'attempts/K7_seed113'
            durable.mkdir(parents=True,exist_ok=False)  # Permanent claim, even if the next write fails.
            path=self.runroot/'K7'/('seed_113_attempt_'+session)
            registration=dict(schema='colab_k7_single_v1',decision='D-021',**IDENTITY,
                primary_identity='K7_seed113',owner='Colab',execution='GPU',start_epoch=1,model_resume=False,
                head=self.package['head'],package_sha256=self.package_hash,readiness_sha256=reviewed_sha,
                output=str(path),log=str(self.session/'K7_seed113.log'),session=session,
                windows_seed503_dependency=False,created_utc=stamp())
            write_new(durable/'REGISTRATION.json',registration)
            require(read(durable/'REGISTRATION.json')==registration, 'Registration readback failed')
            active=dict(registration_sha256=sha(durable/'REGISTRATION.json'),session=session,identity=IDENTITY,
                        confirmation=confirmation,processes=process_snapshot(),utc=stamp())
            write_new(self.durable/'SINGLE_ACTIVE.json',active)
            write_new(self.session/'invocation.json',active)
            def persist():
                save_snapshot(path,durable)
                save_snapshot(self.session,self.durable/'sessions'/session)
            try:
                self.call('train',[7,113,path,self.dataset],'K7_seed113',113,monitor=persist)
                self.integrity()
                result=self.validate(7,113,path,'validate_K7_seed113')
                entries=save_snapshot(path,durable)
                require(entries==inventory(path), 'Final Drive snapshot incomplete')
                receipt=dict(status='VERIFIED_DURABLE_COMPLETE',identity=IDENTITY,files=entries,
                    registration_sha256=sha(durable/'REGISTRATION.json'),validation=result,utc=stamp())
                verify_durable(durable,receipt)
                archive=archive_unit(self.durable/'archives','K7_seed113_'+session,
                    dict(run=path,session=self.session,registration=durable/'REGISTRATION.json'),entries)
                receipt.update(archive=str(archive),archive_sha256=sha(archive))
                write_new(durable/'DURABLE_COMPLETE.json',receipt)
                require(read(durable/'DURABLE_COMPLETE.json')==receipt, 'Completion readback failed')
                persist()
                require(read(self.durable/'SINGLE_ACTIVE.json')==active, 'Launch ownership record changed')
                closed=self.durable/'closed_single_runs';closed.mkdir(exist_ok=True)
                (self.durable/'SINGLE_ACTIVE.json').rename(closed/(session+'.json'))
                return dict(status='ONE_RUN_VERIFIED_DURABLE_COMPLETE',identity=IDENTITY,
                    registration=str(durable/'REGISTRATION.json'),receipt=str(durable/'DURABLE_COMPLETE.json'),
                    receipt_sha256=sha(durable/'DURABLE_COMPLETE.json'),next_run_launched=False)
            except BaseException as exc:
                write_new(self.session/'STOPPED.json',dict(error=repr(exc),utc=stamp(),automatic_retry=False))
                save_snapshot(self.session,self.durable/'sessions'/session)
                raise
