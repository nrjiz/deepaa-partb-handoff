"""Single-run orchestration fixtures; no TensorFlow/GPU/training execution."""
import contextlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from research.code import colab_k7_single as single
from research.code.workflow_evidence import write_new,read,sha,hashes
from research.code import colab_transfer as transfer


@pytest.fixture
def operation(tmp_path,monkeypatch):
    payload=tmp_path/'payload';payload.mkdir()
    write_new(payload/'PACKAGE.json',dict(head='a'*40,release=True,source_status='',preserved_seed503={}))
    write_new(payload/'DATASET.json',{})
    q=single.K7Single(payload,tmp_path/'workspace',tmp_path/'durable',tmp_path/'data')
    q.workspace.mkdir();q.durable.mkdir()
    original=q.runroot/'K5/seed_503';original.mkdir(parents=True);(original/'.RUNNING').write_text('interrupted')
    monkeypatch.setattr(q,'integrity',lambda:None)
    @contextlib.contextmanager
    def lock():yield
    monkeypatch.setattr(single,'local_lock',lock)
    monkeypatch.setattr(single,'verify_images',lambda *a:None)
    monkeypatch.setattr(single,'process_snapshot',lambda:[])
    original_read=Path.read_text
    def text(path,*a,**kw):
        if path.as_posix()=='/proc/sys/kernel/random/boot_id':return 'fixture-boot'
        return original_read(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',text)
    probe=tmp_path/'probe.json';write_new(probe,{})
    ready=tmp_path/'READINESS.json'
    write_new(ready,dict(status='READY_FOR_SENIOR_REVIEW_NOT_LAUNCH',identity=single.IDENTITY,
        head='a'*40,package_sha256=q.package_hash,runtime_boot_id='fixture-boot',
        dataset_inventory_sha256=sha(payload/'DATASET.json'),dataset=str(q.dataset),durable=str(q.durable),
        destination_probe=str(probe),destination_probe_sha256=sha(probe),environment=dict(freeze=['fixture'])))
    return q,ready


@pytest.mark.parametrize('name',['QUEUE_ACTIVE.json','SINGLE_ACTIVE.json','attempts/K7_seed113/REGISTRATION.json'])
def test_existing_claim_blocks_without_worker(operation,monkeypatch,name):
    q,ready=operation
    p=q.durable/name;p.parent.mkdir(parents=True,exist_ok=True);write_new(p,{})
    monkeypatch.setattr(q,'call',lambda *a,**kw:pytest.fail('must not spawn'))
    with pytest.raises(ValueError):q.run(ready,sha(ready),single.CONFIRMATION)
    assert read(p)=={}


def test_existing_completed_local_seed_blocks(operation):
    q,_=operation
    p=q.runroot/'K7/seed_113';p.mkdir(parents=True);write_new(p/'COMPLETED.json',{})
    with pytest.raises(ValueError,match='output already exists'):q.unclaimed()


def test_other_failed_local_attempt_without_running_marker_blocks(operation):
    q,_=operation
    p=q.runroot/'K7/seed_211';p.mkdir(parents=True);write_new(p/'FAILED.json',{})
    with pytest.raises(ValueError,match='Unregistered local output'):q.unclaimed()


def test_success_is_one_run_and_relaunch_blocked(operation,monkeypatch):
    q,ready=operation;calls=[]
    def call(mode,args,label,*a,**kw):
        calls.append(mode)
        if mode=='environment':write_new(args[0],dict(freeze=['fixture']))
        elif mode=='train':
            assert args[:2]==[7,113]
            args[2].mkdir(parents=True);write_new(args[2]/'COMPLETED.json',dict(fixture=True))
        else:pytest.fail('Unexpected dependency: '+mode)
    monkeypatch.setattr(q,'call',call)
    monkeypatch.setattr(q,'validate',lambda k,s,p,l:dict(verdict='PASS_WITH_LIMITATIONS',K=k,seed=s,run=str(p),hashes=hashes(p)))
    def archive(folder,label,sources,identity):
        folder.mkdir();p=folder/'fixture.zip';p.write_bytes(b'fixture');return p
    monkeypatch.setattr(single,'archive_unit',archive)
    result=q.run(ready,sha(ready),single.CONFIRMATION)
    assert result['status']=='ONE_RUN_VERIFIED_DURABLE_COMPLETE'
    assert calls==['environment','train'] and not result['next_run_launched']
    r=read(q.durable/'attempts/K7_seed113/REGISTRATION.json')
    assert r['latent_dimension']==6 and not r['windows_seed503_dependency'] and not r['model_resume']
    assert not (q.durable/'SINGLE_ACTIVE.json').exists()
    with pytest.raises(ValueError,match='already claimed'):q.run(ready,sha(ready),single.CONFIRMATION)
    assert calls==['environment','train']


def test_failed_training_retains_claim_and_never_advances(operation,monkeypatch):
    q,ready=operation;calls=[]
    def call(mode,args,label,*a,**kw):
        calls.append(mode)
        if mode=='environment':write_new(args[0],dict(freeze=['fixture']))
        elif mode=='train':raise RuntimeError('fixture exit 7')
    monkeypatch.setattr(q,'call',call)
    with pytest.raises(RuntimeError,match='exit 7'):q.run(ready,sha(ready),single.CONFIRMATION)
    assert calls==['environment','train']
    assert (q.durable/'SINGLE_ACTIVE.json').is_file()
    assert (q.durable/'attempts/K7_seed113/REGISTRATION.json').is_file()


def test_readiness_checks_only_selected_k7_and_records_destination(operation,monkeypatch):
    q,_=operation;calls=[]
    def call(mode,args,label,*a,**kw):
        calls.append(mode)
        if mode=='data':write_new(args[-1],dict(status='fixture-data'))
        elif mode=='environment':write_new(args[-1],dict(freeze=['fixture']))
        elif mode=='benchmark':
            assert args[:2]==[7,113]
            write_new(args[-1],dict(fingerprint='same',scientific_evidence=False))
        else:pytest.fail('Readiness must not train or depend on K5')
    monkeypatch.setattr(q,'call',call)
    result=q.readiness()
    assert calls==['data','environment','benchmark','benchmark']
    assert result['identity']==single.IDENTITY and not result['windows_seed503_dependency']
    assert sha(Path(result['destination_probe']))==result['destination_probe_sha256']
    assert sha(Path(result['path']))==result['sha256']
    assert not (q.durable/'attempts').exists()


def test_identity_requires_frozen_latent6(operation):
    q,_=operation
    q.package['execution_scope']=dict(**single.IDENTITY,mode='one manual run',owner='Colab')
    config=q.root/'research/configs/archetype_stability_v1.json'
    config.parent.mkdir(parents=True)
    write_new(config,dict(planned_runs=[dict(K=7,seed=113,latent_dimension=4)],
                          primary_scope=dict(latent_dimension_rule='K - 1')))
    with pytest.raises(ValueError,match='Frozen K7 identity differs'):q.identity()


@pytest.mark.parametrize('args',[['--k','5','--latent-dimension','4','--seed','503'],
                               ['--k','7','--latent-dimension','4','--seed','113'],
                               ['--k','7','--latent-dimension','6','--seed','211']])
def test_cli_rejects_unfrozen_or_unapproved_identity(tmp_path,monkeypatch,args):
    tool=Path(__file__).resolve().parents[1]/'tools/colab_k7_single.py'
    command=[str(tool),'launch']
    for n in ('payload','workspace','durable','dataset','result-output'):command+=['--'+n,str(tmp_path)]
    monkeypatch.setattr(sys,'argv',command+args)
    with pytest.raises(SystemExit) as exc:runpy.run_path(str(tool),run_name='__main__')
    assert exc.value.code==2


def test_snapshot_attributes_preserve_mixed_notebook(tmp_path):
    (tmp_path/'mixed.ipynb').write_bytes(b'{\r\n}\n')
    (tmp_path/'lf.py').write_bytes(b'pass\n')
    before=transfer.inventory(tmp_path)
    rules=transfer.snapshot_attributes(tmp_path)
    assert '"mixed.ipynb" text eol=crlf' in rules and '"lf.py" text eol=lf' in rules
    assert transfer.inventory(tmp_path)==before


def test_notebook_cached_import_check_precedes_research_import():
    root=Path(__file__).resolve().parents[2]
    notebook=read(root/'research/notebooks/stability_v1_colab.ipynb')
    setup=next(c for c in notebook['cells'] if c['cell_type']=='code')
    prefix=''.join(setup['source']).split('from pathlib import Path')[0]
    with pytest.raises(AssertionError,match='Cached research imports'):exec(prefix,{})
    all_code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert "not PAYLOAD.exists() and not WORKSPACE.exists()" in all_code
    assert 'FROZEN_K7_LATENT6_CONFIRMED = False' in all_code
    assert "'--latent-dimension', '6'" in all_code
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
