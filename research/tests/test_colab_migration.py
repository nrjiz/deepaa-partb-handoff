"""D-020 fixtures only: no TensorFlow import, GPU or local model training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from research.code import colab_runtime as runtime
from research.code import colab_transfer as transfer
from research.code.workflow_evidence import write_new, read, hashes, sha


def test_real_release_checkout_and_restore(tmp_path):
    """Actual senior release, never a fabricated scientific receipt or model run."""
    source = Path(__file__).resolve().parents[2]
    release = Path(os.environ.get('DEEPAA_TEST_RELEASE', str(source.parent/
        'colab_migration_release_bytecode_fix_20260922T070700040114Z')))
    if not (release/'TRANSFER_RECEIPT.json').exists():
        pytest.skip('Actual release unavailable; restore itself runs these gates in Colab')
    receipt = read(release/'TRANSFER_RECEIPT.json')
    payload = tmp_path/'payload'
    package = transfer.extract_verified(release/receipt['archive'], receipt['sha256'], payload)
    assert sha(payload/'PACKAGE.json') == receipt['package_manifest_sha256']
    assert package['head'] == receipt['head']
    before = transfer.inventory(payload)
    workspace = tmp_path/'workspace'
    if sys.platform.startswith('linux'):
        root = transfer.restore(payload, workspace)
        assert read(workspace/'RESTORE.json')['d019_registration_receipt'] == 'VERIFIED'
    else:
        root = transfer.checkout_source(payload, workspace)
    assert transfer.inventory(payload) == before
    assert not transfer.git(root, 'status', '--porcelain=v1', '--untracked-files=all')
    transfer.verify_index(root, receipt['head'])
    for name, entry in package['files'].items():
        if name.startswith('source_bytes/'):
            assert sha(root/name.removeprefix('source_bytes/')) == entry['sha256']
    # Demonstrate the reported failure without touching the successful checkout.
    old = tmp_path/'blanket_crlf'
    subprocess.run(['git','clone','--no-checkout',str(payload/'history.bundle'),str(old)],check=True)
    for key, value in (('core.autocrlf','true'),('core.eol','crlf')):
        subprocess.run(['git','config',key,value],cwd=old,check=True)
    subprocess.run(['git','checkout','--detach',receipt['head']],cwd=old,check=True)
    index = transfer.verify_index(old, receipt['head'])
    for p in (payload/'source_bytes').rglob('*'):
        if p.is_file():
            (old/p.relative_to(payload/'source_bytes')).write_bytes(p.read_bytes())
    assert transfer.git(old,'status','--porcelain=v1','--untracked-files=all')
    assert not transfer.git(old,'diff','--stat')
    assert transfer.verify_index(old, receipt['head']) == index
    # A true edit must still fail the source guard; no broad partial/dirty bypass.
    (root/'research/COLAB_MIGRATION.md').write_bytes(b'not the reviewed source\n')
    with pytest.raises(ValueError, match='Dirty source/index'):
        runtime.source_guard(root, package)


def test_visible_subprocess_failure_keeps_output(tmp_path, capsys):
    log = tmp_path/'failure.log'
    with pytest.raises(subprocess.CalledProcessError) as failure:
        transfer.run_visible([sys.executable,'-B','-u','-c',
            "import sys; print('progress'); print('failure detail',file=sys.stderr); sys.exit(7)"], log=log)
    assert failure.value.returncode == 7
    assert 'progress' in log.read_text() and 'failure detail' in log.read_text()
    assert 'failure detail' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        transfer.run_visible([sys.executable,'-c','pass'],log=log)


def test_cli_result_is_separate_from_diagnostics(tmp_path, monkeypatch, capsys):
    import runpy
    class FixtureQueue:
        def __init__(self, *args): pass
        def readiness(self):
            print('fixture progress', file=sys.stderr)
            return dict(path='fixture/READINESS.json', sha256='fixture')
    monkeypatch.setattr(runtime, 'ColabQueue', FixtureQueue)
    output = tmp_path/'result.json'
    tool = Path(__file__).resolve().parents[1]/'tools/colab_migration.py'
    monkeypatch.setattr(sys, 'argv', [str(tool), 'readiness', '--payload', str(tmp_path),
        '--workspace', str(tmp_path), '--durable', str(tmp_path), '--dataset', str(tmp_path),
        '--result-output', str(output)])
    runpy.run_path(str(tool), run_name='__main__')
    assert read(output) == dict(path='fixture/READINESS.json', sha256='fixture')
    assert 'fixture progress' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        runpy.run_path(str(tool), run_name='__main__')


def test_fresh_notebook_import_preserves_extracted_payload_inventory(tmp_path):
    root=Path(__file__).resolve().parents[2]
    notebook=read(root/'research/notebooks/stability_v1_colab.ipynb')
    setup=next(c for c in notebook['cells'] if c['cell_type']=='code')['source']
    index=next(i for i,line in enumerate(setup) if line.startswith('import ') and 'sys' in line and 'os' in line)
    assert setup[index+1:index+3]==[
        'sys.dont_write_bytecode = True\n',
        "os.environ['PYTHONDONTWRITEBYTECODE'] = '1'\n",
    ]
    bootstrap=''.join(setup[index:index+3])
    # Extract actual transfer code and its three local dependencies, never pyc files.
    archive=tmp_path/'transfer_fixture.zip'
    files={}
    with zipfile.ZipFile(archive,'w') as z:
        for name in ('__init__.py','colab_transfer.py','workflow_evidence.py','constants.py'):
            data=(root/'research/code'/name).read_bytes()
            member='source_bytes/research/code/'+name
            files[member]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            z.writestr(member,data)
        z.writestr('PACKAGE.json',json.dumps(dict(files=files)))
    payload=tmp_path/'payload'
    transfer.extract_verified(archive,sha(archive),payload)
    before=transfer.inventory(payload)
    env=dict(os.environ)
    for name in ('PYTHONDONTWRITEBYTECODE','PYTHONPYCACHEPREFIX','PYTHONPATH'):
        env.pop(name,None)
    # No -B: the child must begin with bytecode enabled, unlike the pytest parent.
    code='''
import sys, os
from pathlib import Path
import hashlib
assert not sys.dont_write_bytecode
assert 'PYTHONDONTWRITEBYTECODE' not in os.environ
payload=Path(sys.argv[1])
def inventory():
    return {p.relative_to(payload).as_posix():
            (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            if p.is_file() else 'directory' for p in payload.rglob('*')}
before=inventory()
exec(sys.argv[2])
assert sys.dont_write_bytecode and os.environ['PYTHONDONTWRITEBYTECODE']=='1'
assert 'research.code.colab_transfer' not in sys.modules
sys.path.insert(0,str(payload/'source_bytes'))
from research.code import colab_transfer
assert Path(colab_transfer.__file__).resolve()==(payload/'source_bytes/research/code/colab_transfer.py').resolve()
assert inventory()==before, 'Import mutated extracted payload'
assert not list(payload.rglob('__pycache__')) and not list(payload.rglob('*.pyc'))
print('fresh-process payload inventory unchanged')
'''
    result=subprocess.run([sys.executable,'-c',code,str(payload),bootstrap],cwd=tmp_path,
                          env=env,capture_output=True,text=True,check=True)
    assert 'fresh-process payload inventory unchanged' in result.stdout
    assert transfer.inventory(payload)==before


def test_snapshot_roundtrip_and_corruption(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'.RUNNING').write_text('original')
    (run/'training_history.csv').write_text('epoch\n1\n')
    durable=tmp_path/'drive'
    before=hashes(run)
    entries=runtime.save_snapshot(run,durable)
    assert hashes(run)==before
    runtime.verify_durable(durable,dict(files=entries))
    runtime.restore_objects(durable,entries,tmp_path/'restored')
    assert hashes(tmp_path/'restored')==before
    with pytest.raises(ValueError,match='merge'):
        runtime.restore_objects(durable,entries,tmp_path/'restored')
    (durable/'objects'/entries['.RUNNING']['sha256']).write_text('bad')
    with pytest.raises(ValueError,match='hash mismatch'):
        runtime.verify_durable(durable,dict(files=entries))


def test_snapshot_growth_retains_old_bytes(tmp_path):
    run=tmp_path/'run';run.mkdir();log=run/'log';log.write_text('epoch1')
    first=runtime.save_snapshot(run,tmp_path/'drive')
    log.write_text('epoch1\nepoch2')
    second=runtime.save_snapshot(run,tmp_path/'drive')
    assert first!=second
    runtime.verify_durable(tmp_path/'drive',dict(files=first))
    runtime.verify_durable(tmp_path/'drive',dict(files=second))
    assert len(list((tmp_path/'drive/snapshots').glob('*.json')))==2


def test_source_draft_cannot_execute(tmp_path):
    with pytest.raises(ValueError,match='Draft'):
        runtime.source_guard(tmp_path,dict(release=False,source_status='new code'))


def test_preservation_keeps_originals_and_refuses_overwrite(tmp_path):
    root=tmp_path/'repo';state=tmp_path/'state'
    run=root/'research/runs/stability_v1/K5/seed_503';run.mkdir(parents=True)
    (run/'.RUNNING').write_text('original marker')
    (run/'configuration.json').write_text('{"seed":503}')
    (state/'training_logs').mkdir(parents=True)
    (state/'training_logs/K5_seed503_20260915T181345388069Z.log').write_text('epoch9')
    for session in ('20260915T181151848061Z','20260915T133418789705Z'):
        p=state/'sessions'/session;p.mkdir(parents=True);(p/'events.jsonl').write_text('evidence')
    (state/'approved_pins').mkdir()
    (state/'approved_pins/4a5daa34f68d9a68b3b024bb87c7c9ff5c85ccc2.json').write_text('{}')
    (state/'workflow.lock.json').write_text('{"pid":14100}')
    (state/'approved-head.txt').write_text('original pin')
    before=(hashes(root),hashes(state))
    receipt=transfer.preserve(root,state,tmp_path/'preserved')
    assert receipt['original_run_hashes']==hashes(run)
    assert before==(hashes(root),hashes(state))
    for item in receipt['files']:
        assert sha(tmp_path/'preserved'/item['copy'])==item['sha256']==sha(Path(item['source']))
    with pytest.raises(FileExistsError):transfer.preserve(root,state,tmp_path/'preserved')
    assert before==(hashes(root),hashes(state))


@pytest.mark.parametrize('name',['../escape','/absolute','C:/escape','a\\escape'])
def test_zip_path_escape_refused(tmp_path,name):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:
        entry=zipfile.ZipInfo('placeholder');entry.filename=name
        z.writestr(entry,'bad')
    with pytest.raises(ValueError,match='Unsafe'):
        transfer.extract_verified(p,sha(p),tmp_path/'output')


def test_extract_exact_set_and_hash(tmp_path):
    p=tmp_path/'good.zip';data=b'bytes'
    m=dict(files={'source':dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())})
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('source',data);z.writestr('PACKAGE.json',json.dumps(m))
    assert transfer.extract_verified(p,sha(p),tmp_path/'out')==m
    with pytest.raises(ValueError,match='destination exists'):
        transfer.extract_verified(p,sha(p),tmp_path/'out')
    with pytest.raises(ValueError,match='SHA-256'):
        transfer.extract_verified(p,'0'*64,tmp_path/'out2')


def queue_fixture(tmp_path,monkeypatch):
    payload=tmp_path/'payload';payload.mkdir()
    write_new(payload/'PACKAGE.json',dict(release=True,head='a'*40,source_status='',preserved_seed503={}))
    write_new(payload/'DATASET.json',{})
    workspace=tmp_path/'work';(workspace/'aliases').mkdir(parents=True)
    durable=tmp_path/'drive';durable.mkdir()
    q=runtime.ColabQueue(payload,workspace,durable,tmp_path/'data')
    write_new(payload/'RELOCATION.json',{})
    write_new(workspace/'RESTORE.json',dict(package_sha256=q.package_hash,relocation_sha256=sha(payload/'RELOCATION.json')))
    monkeypatch.setattr(q,'guard',lambda:None)
    @runtime.contextlib.contextmanager
    def lock(): yield
    monkeypatch.setattr(runtime,'local_lock',lock)
    monkeypatch.setattr(runtime,'process_snapshot',lambda:[])
    monkeypatch.setattr(runtime,'verify_images',lambda *a:None)
    ready=tmp_path/'READINESS.json'
    # Fixture patches Linux boot-id access only; never a real runtime claim.
    original=Path.read_text
    def read_text(path,*a,**kw):
        if path.as_posix()=='/proc/sys/kernel/random/boot_id':return 'fixture-boot'
        if path.as_posix()=='/proc/meminfo':return 'MemAvailable: 8388608 kB\n'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read_text)
    write_new(ready,dict(package_sha256=q.package_hash,head='a'*40,runtime_boot_id='fixture-boot',environment=dict(freeze=['pinned'])))
    return q,ready


def test_confirmation_and_readiness_hash_required(tmp_path,monkeypatch):
    q,ready=queue_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='declaration'):
        q.run(ready,sha(ready),'yes')
    with pytest.raises(ValueError,match='reviewed'):
        q.run(ready,'0'*64,runtime.CONFIRMATION)
    assert not (q.durable/'QUEUE_ACTIVE.json').exists()


def test_unresolved_queue_blocks_without_takeover(tmp_path,monkeypatch):
    q,ready=queue_fixture(tmp_path,monkeypatch)
    write_new(q.durable/'QUEUE_ACTIVE.json',dict(old='untouched'))
    before=hashes(q.durable)
    with pytest.raises(ValueError,match='Unresolved queue'):
        q.run(ready,sha(ready),runtime.CONFIRMATION)
    assert hashes(q.durable)==before


def test_unresolved_attempt_never_launches(tmp_path,monkeypatch):
    q,ready=queue_fixture(tmp_path,monkeypatch)
    (q.durable/'attempts/K5_seed503').mkdir(parents=True)
    seen=[]
    def call(mode,args,label,*a,**kw):
        seen.append(mode)
        if mode=='environment':write_new(args[0],dict(freeze=['pinned']))
    monkeypatch.setattr(q,'call',call)
    monkeypatch.setattr(q,'validate',lambda k,s,p,l:dict(K=k,seed=s,run=str(p)))
    with pytest.raises(ValueError,match='unresolved attempt'):
        q.run(ready,sha(ready),runtime.CONFIRMATION)
    assert seen==[]
    assert (q.durable/'attempts/K5_seed503').exists()
    assert not (q.durable/'QUEUE_ACTIVE.json').exists()


def test_later_unresolved_attempt_blocks_first_missing_seed(tmp_path,monkeypatch):
    q,ready=queue_fixture(tmp_path,monkeypatch)
    (q.durable/'attempts/K7_seed503').mkdir(parents=True)
    monkeypatch.setattr(q,'call',lambda *a,**kw:pytest.fail('Must not spawn any worker'))
    with pytest.raises(ValueError,match='K7_seed503'):
        q.run(ready,sha(ready),runtime.CONFIRMATION)
    assert not (q.durable/'QUEUE_ACTIVE.json').exists()


def test_failed_child_stops_before_k7_and_retains_claim(tmp_path,monkeypatch):
    q,ready=queue_fixture(tmp_path,monkeypatch)
    seen=[]
    def call(mode,args,label,*a,**kw):
        seen.append((mode,label))
        if mode=='environment':write_new(args[0],dict(freeze=['pinned']))
        if mode=='train':raise RuntimeError('fixture child exit 7')
    monkeypatch.setattr(q,'call',call)
    monkeypatch.setattr(q,'validate',lambda k,s,p,l:dict(K=k,seed=s,run=str(p)))
    with pytest.raises(RuntimeError,match='exit 7'):
        q.run(ready,sha(ready),runtime.CONFIRMATION)
    assert [v for v in seen if v[0]=='train']==[('train','K5_seed503')]
    reg=read(q.durable/'attempts/K5_seed503/REGISTRATION.json')
    assert reg['start_epoch']==1 and not reg['model_resume']
    assert reg['predecessor']['classification']=='INTERRUPTED_NO_RESUMABLE_CHECKPOINT'
    assert (q.durable/'QUEUE_ACTIVE.json').exists()


def test_original_marker_only_exception(tmp_path,monkeypatch):
    q,_=queue_fixture(tmp_path,monkeypatch)
    # Call the real guard rather than the fixture stub.
    monkeypatch.setattr(runtime,'source_guard',lambda *a:None)
    monkeypatch.setattr(runtime,'no_competing',lambda:[])
    monkeypatch.setattr(runtime.shutil,'disk_usage',lambda p:type('Disk',(),{'free':10*1024**3})())
    original=q.runroot/'K5/seed_503';original.mkdir(parents=True)
    (original/'.RUNNING').write_text('preserve')
    q.package['preserved_seed503']=hashes(original)
    runtime.ColabQueue.guard(q)
    extra=q.runroot/'K7/seed_113';extra.mkdir(parents=True);(extra/'.RUNNING').write_text('active')
    with pytest.raises(ValueError,match='Unresolved local'):
        runtime.ColabQueue.guard(q)
    (original/'unexpected').write_text('changed')
    with pytest.raises(ValueError,match='predecessor changed'):
        runtime.ColabQueue.guard(q)


def test_frozen_queue_and_notebook_disabled():
    assert runtime.QUEUE==[(5,503),(7,113),(7,211),(7,307),(7,401),(7,503)]
    root=Path(__file__).resolve().parents[2]
    nb=read(root/'research/notebooks/stability_v1_colab.ipynb')
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')
    launches=[c for c in nb['cells'] if c['cell_type']=='code' and 'PRIMARY TRAINING' in ''.join(c['source'])]
    assert len(launches)==1 and 'PRIMARY_LAUNCH_ENABLED = False' in ''.join(launches[0]['source'])


def test_six_runs_sequential_durable_before_next_and_single_summary_identity(tmp_path,monkeypatch):
    q,ready=queue_fixture(tmp_path,monkeypatch)
    seen=[]
    for s in (113,211,307,401):
        p=q.old_path(5,s);p.mkdir(parents=True);(p/'artifact').write_text(str(s))
    def validate(k,s,path,label):
        return dict(verdict='PASS_WITH_LIMITATIONS',K=k,seed=s,run=str(path),hashes=hashes(path),execution_revision='a'*40)
    monkeypatch.setattr(q,'validate',validate)
    def call(mode,args,label,*a,**kw):
        if mode=='environment':write_new(args[0],dict(freeze=['pinned']))
        if mode=='train':
            for prior_k,prior_s in seen:
                assert (q.durable/f'attempts/K{prior_k}_seed{prior_s}/DURABLE_COMPLETE.json').exists()
            k,s,path,_=args
            seen.append((k,s));path.mkdir(parents=True);(path/'artifact').write_text(f'{k}/{s}')
        if mode=='compare':
            args[2].mkdir(parents=True);(args[2]/'artifact').write_text('pair')
        if mode=='pair_verify':
            pair,ref,cand,k,s,out=args
            pair.mkdir(parents=True,exist_ok=True)
            write_new(out,dict(verdict='PASS',K=k,seed=s,pair=str(pair),hashes=hashes(pair),
                              inputs=dict(reference=hashes(ref),candidate=hashes(cand))))
        if mode=='aggregate':
            value=read(args[0]);assert len(value['runs'])==10
            assert len({(r['K'],r['seed']) for r in value['runs']})==10
            args[1].mkdir(parents=True);write_new(args[1]/'summary.json',dict(status='COMPLETE'))
    monkeypatch.setattr(q,'call',call)
    import research.code.workflow_evidence as evidence
    def archive(destination,label,sources,identity):
        destination.mkdir(parents=True,exist_ok=True)
        target=destination/(label+'.zip');target.write_bytes(b'fixture_archive');return target
    monkeypatch.setattr(evidence,'archive_unit',archive)
    result=q.run(ready,sha(ready),runtime.CONFIRMATION)
    assert seen==runtime.QUEUE and result['status']=='COMPLETE'
    assert not (q.durable/'QUEUE_ACTIVE.json').exists()
    audit=read(Path(result['bundle'])/'ATTEMPT_AUDIT.json')
    assert len(audit['attempts'])==6 and audit['attempts'][0]['predecessor'] is not None
    # Completed durable attempts are validated and reused, never trained again.
    result2=q.run(ready,sha(ready),runtime.CONFIRMATION)
    assert result2['status']=='COMPLETE' and seen==runtime.QUEUE


@pytest.mark.parametrize('mutation',[{'seed':401},{'attempt_name':'../escape'},{'start_epoch':9},{'model_resume':True}])
def test_registration_rejects_identity_path_and_resume(mutation):
    r=dict(K=5,latent_dimension=4,seed=503,primary_identity='K5_seed503',decision='D-020',package_sha256='p',head='h',
           start_epoch=1,model_resume=False,attempt_name='seed_503_attempt_20260921T120000123456Z_0123abcd',
           predecessor=dict(path='research/runs/stability_v1/K5/seed_503',classification='INTERRUPTED_NO_RESUMABLE_CHECKPOINT',hashes={}))
    runtime.verify_registration(r,5,503,'p','h',{})
    r.update(mutation)
    with pytest.raises(ValueError):runtime.verify_registration(r,5,503,'p','h',{})


@pytest.mark.skipif(sys.platform!='linux',reason='Linux flock and proc must also run in Colab readiness')
def test_real_linux_duplicate_lock_and_process_identity(tmp_path):
    with runtime.local_lock(tmp_path/'lock'):
        with pytest.raises(ValueError,match='owns the lock'):
            with runtime.local_lock(tmp_path/'lock'):pass
    rows=runtime.process_snapshot()
    assert all(set(('pid','ppid','start_ticks','boot_id','executable','command'))<=set(r) for r in rows)


def test_d019_bound_sources_unchanged():
    root=Path(__file__).resolve().parents[2]
    from research.code.recovery import code_hashes
    receipt=root.parent/'stability_workflow_local/recovered_exports/K5_seed401_20260915/RECOVERED_EXPORT.json'
    if not receipt.exists():
        # In Colab the actual receipt is exercised by readiness, not a fabricated path.
        pytest.skip('Original Windows receipt checked by real target readiness')
    assert code_hashes(root)==read(receipt)['code_hashes']
