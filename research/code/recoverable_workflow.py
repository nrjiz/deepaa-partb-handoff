"""Proposed Route B governance. Draft packages cannot start or resume primary work."""
import os,sys,uuid,subprocess
from pathlib import Path
from .workflow_evidence import read,write_new,sha,require,archive_unit
from .recoverable_checkpoint import EpochStore,verify,transfer
from .recoverable_model import VARIANT
DURABLE=Path('/content/drive/MyDrive/DeepAA_stability_v1_D020')
CONFIRM='I confirm all previous project runtimes are stopped and only this runtime will own this logical attempt.'

def require_authorization(a,k=7,seed=113):
    require((k,seed)==(7,113),'Only K7/latent6/seed113')
    require(a.get('variant')==VARIANT and a.get('scientific_approval')=='ACCEPTED_PROSPECTIVELY','Scientific amendment not approved')
    require(a.get('action') in ('start','continue'),'Explicit action required')
    require(a.get('epochs')==50 and a.get('latent_dimension')==6,'Frozen budget/identity')
    require(a.get('single_runtime_confirmation')==CONFIRM,'Exclusive manual runtime confirmation required')
    for key in ['scientific_decision_sha256','package_sha256','readiness_sha256']:
        value=a.get(key,'');require(len(value)==64 and all(c in '0123456789abcdef' for c in value),'Missing digest: '+key)
    require(bool(a.get('segment_id')) and a['segment_id'].isalnum(),'New explicit segment ID required')
    if a['action']=='continue':
        require(a.get('parent_segment_id') and a.get('prior_runtime_stopped') is True,'Prior owner review required')
        require(a.get('checkpoint') and len(a.get('checkpoint_receipt_sha256',''))==64,'Explicit reviewed checkpoint required')

def approval_gate(package,auth,package_hash,head):
    require_authorization(auth)
    require(package.get('release') is True and not package.get('source_status'),'Review draft is nonexecutable')
    require(package['head']==head==auth.get('head'),'Reviewed source HEAD mismatch')
    require(auth['package_sha256']==package_hash,'Package authorization mismatch')
    require(package.get('execution_scope')==dict(K=7,latent_dimension=6,seed=113,mode='one manual run',owner='Colab'),'Scope mismatch')

def run(root,payload,workspace,dataset,durable,authorization,reviewed_sha):
    from .colab_runtime import local_lock,no_competing,source_guard,verify_images,save_snapshot
    from .colab_transfer import git
    from research.tools.colab_gpu_worker import gpu_environment
    root,payload,workspace,dataset,durable=map(lambda p:Path(p).resolve(),(root,payload,workspace,dataset,durable))
    require(sha(authorization)==reviewed_sha,'Authorization digest not reviewed')
    auth=read(authorization);package=read(payload/'PACKAGE.json');ph=sha(payload/'PACKAGE.json')
    approval_gate(package,auth,ph,git(root,'rev-parse','HEAD'))
    require(sys.platform.startswith('linux'),'Linux GPU only')
    require(durable==DURABLE and os.path.ismount('/content/drive'),'Fixed mounted Drive required')
    require(os.environ.get('PYTHONHASHSEED')=='113' and os.environ.get('TF_DETERMINISTIC_OPS')=='1' and os.environ.get('MPLBACKEND')=='Agg','Fresh deterministic worker environment required')
    with local_lock():
        no_competing();source_guard(root,package)
        restored=read(workspace/'RESTORE.json');require(restored['package_sha256']==ph,'Restoration identity')
        ready_path=Path(auth['readiness']);require(sha(ready_path)==auth['readiness_sha256'],'Readiness digest')
        ready=read(ready_path)
        require(ready.get('variant')==VARIANT and ready.get('status')=='RECOVERY_GPU_ACCEPTANCE_REVIEWED','Route B target acceptance not reviewed')
        require(ready['package_sha256']==ph and ready['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'Fresh runtime/source readiness required')
        verify_images(dataset,read(payload/'DATASET.json'))
        actual=gpu_environment()
        environment={k:actual[k] for k in ('python','versions','freeze','tensorflow_build','float_policy')}
        environment['gpu_identity']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
        require(environment==ready['environment'],'Environment/hardware differs from reviewed acceptance')
        require(not (durable/'QUEUE_ACTIVE.json').exists(),'Legacy queue claim unresolved')
        attempt=durable/'attempts/K7_seed113';segment=auth['segment_id']
        if auth['action']=='start':
            require(not attempt.exists() and not (durable/'SINGLE_ACTIVE.json').exists(),'Existing attempt/owner; no replacement')
            require(not list((root/'research/runs/stability_v1/K7').glob('seed_113*')),'Existing local attempt')
            attempt.mkdir(parents=True,exist_ok=False)
            write_new(attempt/'REGISTRATION.json',dict(schema='route_b_logical_attempt_v1',variant=VARIANT,K=7,latent_dimension=6,seed=113,epochs=50,head=package['head'],package_sha256=ph,scientific_decision_sha256=auth['scientific_decision_sha256'],original_authorization_sha256=reviewed_sha,model_resume_policy='REVIEWED_EPOCH_BOUNDARY_CHILDREN_ONLY'))
            write_new(durable/'SINGLE_ACTIVE.json',dict(schema='route_b_owner_chain_v1',registration_sha256=sha(attempt/'REGISTRATION.json'),first_segment=segment))
        reg=read(attempt/'REGISTRATION.json')
        require(reg['schema']=='route_b_logical_attempt_v1' and reg['variant']==VARIANT,'Cannot adopt historical attempts')
        require(reg['package_sha256']==ph and reg['head']==package['head'] and reg['scientific_decision_sha256']==auth['scientific_decision_sha256'],'Original registration mismatch')
        owner=read(durable/'SINGLE_ACTIVE.json');require(owner['registration_sha256']==sha(attempt/'REGISTRATION.json'),'Ownership claim mismatch')
        require(not (attempt/'DURABLE_COMPLETE.json').exists(),'Logical attempt already complete')
        identity=dict(variant=VARIANT,registration_sha256=sha(attempt/'REGISTRATION.json'),head=package['head'],package_sha256=ph,dataset_sha256=sha(payload/'DATASET.json'),environment=environment)
        selected=None
        if auth['action']=='continue':
            parent=attempt/'segments'/auth['parent_segment_id'];require((parent/'SEGMENT.json').is_file(),'Unknown parent segment')
            require(not (parent/'SUPERSEDED.json').exists(),'Parent already has a continuation')
            selected=Path(auth['checkpoint']).resolve();require(selected.parent==attempt/'checkpoints','Checkpoint outside registered attempt')
            verify(selected,identity,auth['checkpoint_receipt_sha256'])
            require(checkpoint_ancestor(attempt,auth['parent_segment_id'],read(selected/'ORIGIN.json')['segment_id']),'Checkpoint not from parent lineage')
            write_new(parent/'SUPERSEDED.json',dict(child=segment,authorization_sha256=reviewed_sha,prior_runtime_stopped=True,checkpoint_receipt_sha256=auth['checkpoint_receipt_sha256']))
        record=attempt/'segments'/segment;record.mkdir(parents=True,exist_ok=False)
        write_new(record/'SEGMENT.json',dict(authorization=auth,authorization_sha256=reviewed_sha,registration_sha256=identity['registration_sha256'],boot_id=ready['boot_id']))
        session=workspace/'recovery_segments'/segment;session.mkdir(parents=True,exist_ok=False)
        try:
            if selected:
                selected=transfer(selected,session/'retrieved'/selected.name,identity,auth['checkpoint_receipt_sha256'])
            store=EpochStore(session/'checkpoints',attempt/'checkpoints',identity,selected,auth.get('checkpoint_receipt_sha256'))
            store.segment_id=segment
            from .recoverable_runner import RunSpec,execute_run
            split=root/'research/splits/journal_v1'
            output=root/'research/runs/stability_v1/K7'/('seed_113_route_b_'+segment)
            runpath=execute_run(RunSpec(repository=root,k=7,seed=113,output_dir=output,train_manifest=split/'train.csv',validation_manifest=split/'validation.csv',test_manifest=split/'test.csv',image_dirs=[dataset/f'HAM10000_images_part_{i}' for i in (1,2)]),store,auth)
            write_new(session/'IDENTITY.json',identity)
            subprocess.run([sys.executable,'-B',str(root/'research/tools/validate_recoverable_k7.py'),str(runpath),str(authorization),str(session/'IDENTITY.json'),str(record/'VALIDATION.json')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1'))
            result=read(record/'VALIDATION.json')
            entries=save_snapshot(runpath,record)
            from .colab_transfer import inventory
            require(entries==inventory(runpath),'Final durable snapshot incomplete')
            archive=archive_unit(durable/'archives','route_b_'+segment,dict(run=runpath,segment=record),entries)
            write_new(attempt/'DURABLE_COMPLETE.json',dict(variant=VARIANT,registration_sha256=identity['registration_sha256'],final_segment=segment,validation_sha256=sha(record/'VALIDATION.json'),archive=str(archive),archive_sha256=sha(archive),epochs=50))
            return result
        except BaseException as exc:
            write_new(record/'STOPPED.json',dict(error=repr(exc),automatic_retry=False,policy='Only verified persisted epochs can be reviewed for continuation; no mid-epoch recovery'))
            raise


def checkpoint_ancestor(attempt,parent,origin):
    seen=set()
    while parent:
        require(parent.isalnum() and parent not in seen,'Invalid/cyclic segment lineage');seen.add(parent)
        value=read(attempt/'segments'/parent/'SEGMENT.json')
        if parent==origin:return True
        parent=value['authorization'].get('parent_segment_id')
    return False
