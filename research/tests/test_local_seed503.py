"""D-021 orchestration fixtures only; no real training, lock release or registration."""
from pathlib import Path
import os
import subprocess
import pytest
from research.code import local_seed503 as local
from research.code.workflow_evidence import write_new, read, hashes


@pytest.fixture
def task(tmp_path, monkeypatch):
    t=local.LocalSeed503(tmp_path/'repo',tmp_path/'state','a'*40)
    t.original.mkdir(parents=True)
    (t.original/'.RUNNING').write_text('original')
    monkeypatch.setattr(local,'ORIGINAL',hashes(t.original))
    return t


def test_marker_exception_is_exact_and_other_markers_block(task):
    task.predecessor()
    p=task.w.run_root/'K7/seed_113';p.mkdir(parents=True);(p/'.RUNNING').write_text('other')
    with pytest.raises(ValueError,match='Other active'):task.predecessor()


def test_changed_original_and_existing_sibling_block(task):
    (task.original/'extra').write_text('changed')
    with pytest.raises(ValueError,match='Original'):task.predecessor()


def test_sibling_blocks_even_empty(task):
    task.original.with_name('seed_503_attempt_old').mkdir()
    with pytest.raises(ValueError,match='sibling'):task.predecessor()


def test_register_refuses_existing_folder_before_inspection(task,monkeypatch):
    task.folder.mkdir(parents=True)
    monkeypatch.setattr(task,'inspect',lambda *a:pytest.fail('must stop first'))
    with pytest.raises(ValueError,match='already exists'):task.register(Path('unused'))


def test_existing_launch_claim_blocks_even_without_output(task,monkeypatch):
    task.folder.mkdir(parents=True)
    write_new(task.folder/'LAUNCH.json',{})
    monkeypatch.setattr(task,'registration',lambda:{})
    monkeypatch.setattr(task,'guard',lambda *a:pytest.fail('no second launch'))
    with pytest.raises(ValueError,match='already claimed'):task.launch(local.CONFIRMATION)


def test_missing_confirmation_blocks_before_registration(task,monkeypatch):
    monkeypatch.setattr(task,'registration',lambda:pytest.fail('must not inspect'))
    with pytest.raises(ValueError,match='confirmation'):task.launch('yes')


def test_failed_child_claim_retained_no_k7(task,monkeypatch):
    task.folder.mkdir(parents=True)
    output=task.original.with_name('seed_503_attempt_20260922T120000000000Z')
    record=dict(output=str(output),log=str(task.folder/'train.log'))
    write_new(task.folder/'REGISTRATION.json',record)
    monkeypatch.setattr(task,'registration',lambda:record)
    monkeypatch.setattr(task,'guard',lambda *a:{})
    monkeypatch.setattr(local,'snapshot',lambda:[])
    monkeypatch.setattr(local,'no_competitors',lambda *a:None)
    locks=[]
    class Lock:
        def __init__(self,*a):pass
        def acquire(self):locks.append('acquire')
        def release(self):locks.append('release')
    monkeypatch.setattr(local,'WorkflowLock',Lock)
    calls=[]
    class Child:
        pid=123
        def __init__(self,cmd,**kw):calls.append((cmd,kw))
        def wait(self):return 7
        def poll(self):return 7
    monkeypatch.setattr(local.subprocess,'Popen',Child)
    with pytest.raises(ValueError,match='child failed'):task.launch(local.CONFIRMATION)
    assert locks==['acquire','release'] and len(calls)==1
    cmd,kw=calls[0]
    assert cmd[cmd.index('--k')+1]=='5' and cmd[cmd.index('--seed')+1]=='503'
    assert '--allow-completed-rerun' not in cmd and kw['env']['CUDA_VISIBLE_DEVICES']=='-1'
    assert read(task.folder/'LAUNCH.json')['start_epoch']==1
    assert read(task.folder/'EXIT.json')['exit_code']==7
    assert hashes(task.original)==local.ORIGINAL


def test_unrelated_python_still_blocks():
    rows=[dict(ProcessId=os.getpid(),ParentProcessId=0,Name='python.exe',CommandLine='self'),
          dict(ProcessId=999999,ParentProcessId=0,Name='python.exe',CommandLine='streamlit')]
    with pytest.raises(ValueError,match='Competing process'):local.no_competitors(rows)


@pytest.mark.parametrize('key,value',[('K',7),('seed',401),('start_epoch',9),('model_resume',True),('owner','Colab')])
def test_registration_rejects_identity_or_resume(task,key,value):
    task.folder.mkdir(parents=True)
    r=dict(schema='local_seed503_v1',decision='D-021',K=5,latent_dimension=4,seed=503,
           owner='Windows',start_epoch=1,model_resume=False)
    r[key]=value
    write_new(task.folder/'REGISTRATION.json',r)
    with pytest.raises(ValueError,match='identity/scope'):task.registration()


def test_colab_worker_rejects_k5_before_gpu_or_training(monkeypatch):
    import runpy,sys
    tool=Path(__file__).resolve().parents[1]/'tools/colab_gpu_worker.py'
    monkeypatch.setattr(sys,'platform','linux')
    monkeypatch.setattr(sys,'argv',[str(tool),'train','5','503','unused','unused'])
    with pytest.raises(ValueError,match='Windows-owned'):
        runpy.run_path(str(tool),run_name='__main__')
