"""Engineering contracts only; no scientific training or Drive access."""
import json,shutil
from pathlib import Path
import pytest
from research.code.workflow_evidence import read,write_new,sha
from research.code.recoverable_checkpoint import seal,verify,transfer,EpochStore,rng_state,restore_rng
from research.code.recoverable_workflow import require_authorization,approval_gate,CONFIRM,checkpoint_ancestor
from research.code.recoverable_model import VARIANT
ROOT=Path(__file__).resolve().parents[2]

def auth():return dict(variant=VARIANT,scientific_approval='ACCEPTED_PROSPECTIVELY',action='start',epochs=50,latent_dimension=6,single_runtime_confirmation=CONFIRM,scientific_decision_sha256='a'*64,package_sha256='b'*64,readiness_sha256='c'*64,segment_id='test',head='head')
@pytest.fixture
def checkpoint(tmp_path):
    p=tmp_path/'source'/'identity';p.mkdir(parents=True)
    (p/'state.index').write_bytes(b'synthetic-index');(p/'state.data').write_bytes(b'synthetic-tensors')
    write_new(p/'CONTROL.json',dict(epoch=1,step=251,phase='epoch_complete',next_operation='next_epoch_or_final_export',history=[dict(epoch=1)],prefix='state-1'))
    seal(p,{'identity':'fixture'});return p,{'identity':'fixture'},sha(p/'COMPLETE.json')

def test_transfer_and_readback(checkpoint,tmp_path):
    p,i,h=checkpoint;target=tmp_path/'retrieved'/p.name;transfer(p,target,i,h)
    assert verify(target,i,h)['epoch']==1
    with pytest.raises(FileExistsError):transfer(p,target,i,h)
@pytest.mark.parametrize('fault',['missing','corrupt','receipt','identity','unreviewed','incomplete'])
def test_reject_checkpoint_faults(checkpoint,fault):
    p,i,h=checkpoint
    if fault=='missing':(p/'state.data').rename(p/'saved-missing')
    if fault=='corrupt':(p/'state.data').write_bytes(b'bad')
    if fault=='receipt':(p/'COMPLETE.json').write_text('{}')
    if fault=='identity':i={'identity':'other'}
    if fault=='unreviewed':h='0'*64
    if fault=='incomplete':(p/'COMPLETE.json').rename(p/'UNPUBLISHED.json')
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):verify(p,i,h)
@pytest.mark.parametrize('epoch,step,history',[(0,0,[]),(51,12801,[]),(1,250,[{'epoch':1}]),(2,502,[{'epoch':2}])])
def test_budget_and_history_rejected(tmp_path,epoch,step,history):
    p=tmp_path/'bad';p.mkdir();write_new(p/'CONTROL.json',dict(epoch=epoch,step=step,history=history,phase='epoch_complete',next_operation='next_epoch_or_final_export',prefix='state-1'))
    with pytest.raises(ValueError):seal(p,{})
@pytest.mark.parametrize('key,value',[('scientific_approval','PENDING'),('epochs',100),('latent_dimension',4),('variant','legacy'),('single_runtime_confirmation',''),('segment_id','../bad')])
def test_authorization_rejected(key,value):
    a=auth();a[key]=value
    with pytest.raises(ValueError):require_authorization(a)
@pytest.mark.parametrize('k,seed',[(5,113),(7,211),(7,503)])
def test_identity_rejected(k,seed):
    with pytest.raises(ValueError):require_authorization(auth(),k,seed)
def test_resume_requires_explicit_parent_and_digest():
    a=auth();a['action']='continue'
    with pytest.raises(ValueError):require_authorization(a)
def test_draft_never_executes():
    with pytest.raises(ValueError,match='nonexecutable'):approval_gate({'release':False},auth(),'b'*64,'head')
def test_approved_clean_gate_and_head_mismatch():
    p=dict(release=True,source_status='',head='head',execution_scope=dict(K=7,latent_dimension=6,seed=113,mode='one manual run',owner='Colab'))
    approval_gate(p,auth(),'b'*64,'head')
    with pytest.raises(ValueError):approval_gate(p,auth(),'b'*64,'different')
def test_lineage(tmp_path):
    for child,parent in [('one',None),('two','one')]:
        p=tmp_path/'segments'/child;p.mkdir(parents=True);write_new(p/'SEGMENT.json',{'authorization':{'parent_segment_id':parent}})
    assert checkpoint_ancestor(tmp_path,'two','one')
    assert not checkpoint_ancestor(tmp_path,'one','two')
def test_python_numpy_complete_rng():
    before=rng_state();restore_rng(before);assert rng_state()==before

def test_real_inference_weight_compatibility_without_training(tmp_path):
    import tensorflow as tf,numpy as np
    from research.code.recoverable_model import build_model
    from research.code.model_adapter import build_model as original
    _,model,_=build_model(ROOT,7);x=tf.zeros([1,128,128,3]);model(x,training=False)
    file=tmp_path/'final.weights.h5';model.save_weights(file)
    _,other,_=original(ROOT,7);other(x,training=False);other.load_weights(file)
    assert len(model.weights)==len(other.weights)==30
    for a,b in zip(model.weights,other.weights):np.testing.assert_array_equal(a.numpy(),b.numpy())
    np.testing.assert_array_equal(model(x,training=False)['a'].numpy(),other(x,training=False)['a'].numpy())
    assert not hasattr(model,'observed_samples')

def test_real_checkpoint_roundtrip_no_optimizer_updates(tmp_path):
    import tensorflow as tf,numpy as np
    class Model(tf.Module):
        def __init__(self):
            super().__init__();self.w=tf.Variable([1.,2.]);self.sampling_rng=tf.random.Generator.from_seed(113)
        @property
        def trainable_variables(self):return [self.w]
        def __call__(self,*a,**k):return self.w
    model=Model();opt=tf.keras.optimizers.Adam(.001);opt.build(model.trainable_variables)
    # Synthetic progress fixture only: no claim these251 updates occurred.
    opt.iterations.assign(251);it=iter(tf.data.Dataset.range(3));list(it)
    store=EpochStore(tmp_path/'local',tmp_path/'durable',{'fixture':True});store.segment_id='one'
    record=store.persist(model,opt,it,1,[{'epoch':1}],[])
    expected=model.sampling_rng.normal([4]).numpy();model.w.assign([7.,8.])
    new=Model();newopt=tf.keras.optimizers.Adam(.001);newit=iter(tf.data.Dataset.range(3))
    restored=EpochStore(tmp_path/'other',tmp_path/'durable',{'fixture':True},record['path'],record['receipt_sha256'])
    history,progress=restored.load(new,newopt,newit)
    np.testing.assert_array_equal(new.w.numpy(),[1.,2.]);np.testing.assert_array_equal(new.sampling_rng.normal([4]).numpy(),expected)
    assert int(newopt.iterations)==251 and history==[{'epoch':1}]

def test_persistence_failure_propagates_before_next_epoch(monkeypatch,tmp_path):
    import tensorflow as tf
    import research.code.recoverable_checkpoint as m
    class Model(tf.Module):
        def __init__(self):super().__init__();self.sampling_rng=tf.random.Generator.from_seed(113)
    opt=tf.keras.optimizers.Adam(.001);opt.iterations.assign(251)
    store=EpochStore(tmp_path/'local',tmp_path/'durable',{});store.segment_id='x'
    def fail(*a):raise OSError('simulated Drive interruption')
    monkeypatch.setattr(m,'transfer',fail)
    with pytest.raises(OSError):store.persist(Model(),opt,iter(tf.data.Dataset.range(1)),1,[{'epoch':1}],[])
    assert not store.receipts and not (tmp_path/'durable').exists()

def test_validation_scope_then_numerical_delegate(monkeypatch,tmp_path):
    import research.code.recoverable_validation as m
    a=auth();config=dict(sampling_variant=VARIANT,not_original_frozen_trajectory=True,prospective_amendment=a['scientific_decision_sha256'],continuation=a,implementation_git_commit='head')
    write_new(tmp_path/'configuration.json',config);called=[]
    monkeypatch.setattr(m,'validate_k7_reference',lambda *x:called.append(x) or {'verdict':'PASS_WITH_LIMITATIONS'})
    result=m.validate(ROOT,tmp_path,['head'],a,dict(head='head',registration_sha256='r'))
    assert called and result['trajectory_equivalence_to_original'] is False
    config['sampling_variant']='legacy';(tmp_path/'configuration.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):m.validate(ROOT,tmp_path,['head'],a,dict(head='head'))
from research.tests.test_colab_k7_validation import synthetic_run,HEAD

def test_complete_route_b_artifact_validation(synthetic_run):
    from research.code.recoverable_validation import validate
    run,refresh=synthetic_run;a=auth();a['head']=HEAD
    config=read(run/'configuration.json');config.update(sampling_variant=VARIANT,not_original_frozen_trajectory=True,prospective_amendment=a['scientific_decision_sha256'],continuation=a)
    (run/'configuration.json').write_text(json.dumps(config));refresh()
    result=validate(ROOT,run,[HEAD],a,dict(head=HEAD,registration_sha256='fixture'))
    assert result['verdict']=='PASS_WITH_LIMITATIONS' and result['variant']==VARIANT
    # TensorFlow/model reload is this inherited fixture's test double, not training evidence.

def test_final_epoch_restore_does_not_schedule_more_training():
    source=(ROOT/'research/code/recoverable_runner.py').read_text(encoding='utf-8')
    assert 'range(completed_epochs + 1, epochs + 1)' in source
    assert list(range(50+1,50+1))==[]
    assert 'train_epochs(model=model' in source
    assert source.index('train_epochs(model=model')<source.index('progress.phase("checkpoint_export_and_reload")')
