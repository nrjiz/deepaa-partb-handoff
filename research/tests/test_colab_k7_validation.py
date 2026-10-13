"""ST-04A synthetic schema and complete worker-to-validator regressions.

Fixture checkpoints/model/TF image APIs are test doubles, not trained models.
The actual shared validator, schema functions, arrays, metrics, inventory and
worker validate dispatch execute. No train_step or primary training is called.
"""
from pathlib import Path
from io import StringIO
from types import SimpleNamespace
import sys,runpy
import numpy as np,pandas as pd,pytest
from PIL import Image
from research.code import colab_k7_validation as adapter, workflow_evidence as evidence
from research.code import constants,model_adapter,determinism
from research.code.stability_metrics import semantic_class_rows,semantic_distribution_details,nearest_real_top_n
ROOT=Path(__file__).resolve().parents[2]
HEAD='bac790675e024764763f28ff3d1ec16de2387e1c'


def schema_rows():
    return pd.DataFrame(semantic_class_rows(np.full((2,7),1/7),['nv','mel']))


def compare(a,b):
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-6,atol=1e-12)


def roundtrip(tmp_path,frame):
    path=tmp_path/'semantic.csv';frame.to_csv(path,index=False)
    with adapter.semantic_reason_loader(path):return pd.read_csv(path)


def test_empty_reason_roundtrip_and_original_failure(tmp_path):
    expected=schema_rows()
    with pytest.raises(AssertionError,match='reason'):compare(pd.read_csv(StringIO(expected.to_csv(index=False))),expected)
    actual=roundtrip(tmp_path,expected)
    assert all(x is None for x in actual.reason);compare(actual,expected)


@pytest.mark.parametrize('column,value',[('reason','different'),('reason','NA'),('reason','NaN'),('reason','null'),('reason','None'),('reason',' '),('status','undefined')])
def test_reason_status_mismatch(tmp_path,column,value):
    expected=schema_rows();changed=expected.copy();changed.loc[0,column]=value
    with pytest.raises(AssertionError):compare(roundtrip(tmp_path,changed),expected)


@pytest.mark.parametrize('column',['reason','status','coefficient_mass'])
def test_required_fields(tmp_path,column):
    expected=schema_rows()
    with pytest.raises(AssertionError):compare(roundtrip(tmp_path,expected.drop(columns=column)),expected)


@pytest.mark.parametrize('column',['coefficient_weighted_probability','enrichment_ratio','test_dx_prevalence','coefficient_mass'])
@pytest.mark.parametrize('value',[np.nan,np.inf,.1234567])
def test_numeric_mismatch(tmp_path,column,value):
    expected=schema_rows();changed=expected.copy();i=expected.index[expected[column].notna()][0];changed.loc[i,column]=value
    with pytest.raises(AssertionError):compare(roundtrip(tmp_path,changed),expected)


def test_zero_mass_and_tolerance(tmp_path):
    a=np.full((2,7),1/6);a[:,0]=0
    expected=pd.DataFrame(semantic_class_rows(a,['nv','mel']))
    actual=roundtrip(tmp_path,expected);compare(actual,expected)
    assert actual.loc[actual.status=='undefined','reason'].eq('zero_coefficient_mass').all()
    assert actual.loc[actual.status=='undefined','coefficient_weighted_probability'].isna().all()
    expected=schema_rows();near=expected.copy();near.loc[0,'coefficient_mass']+=1e-8
    compare(roundtrip(tmp_path,near),expected)
    near.loc[0,'coefficient_mass']+=1e-3
    with pytest.raises(AssertionError):compare(roundtrip(tmp_path,near),expected)


def test_exact_path_restore_and_converter_conflict(tmp_path):
    target=tmp_path/'target.csv';other=tmp_path/'other.csv';schema_rows().to_csv(target,index=False);schema_rows().to_csv(other,index=False)
    original=pd.read_csv
    with pytest.raises(RuntimeError):
        with adapter.semantic_reason_loader(target):
            assert all(x is None for x in pd.read_csv(target).reason)
            assert pd.read_csv(other).reason.isna().all()
            with pytest.raises(ValueError):pd.read_csv(target,converters={})
            raise RuntimeError('fixture error')
    assert pd.read_csv is original


@pytest.mark.parametrize('identity',[(5,4,503),(7,4,113),(7,6,211)])
def test_identity_scope(tmp_path,identity):
    evidence.write_new(tmp_path/'configuration.json',dict(zip(['K','latent_dimension','seed'],identity)))
    with pytest.raises(ValueError,match='only K7'):adapter.validate_k7_reference(ROOT,tmp_path,[HEAD])


def test_recovered_path_refused(tmp_path):
    evidence.write_new(tmp_path/'RECOVERED_EXPORT.json',{})
    with pytest.raises(ValueError,match='recovered'):adapter.validate_k7_reference(ROOT,tmp_path,[HEAD])


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    def numpy(self):return self.value


def to_uint8(value,*a,**kw):return Tensor((np.asarray(value)*255).astype(np.uint8))


@pytest.fixture
def synthetic_run(tmp_path,monkeypatch):
    run=tmp_path/'SYNTHETIC_NOT_SCIENTIFIC';run.mkdir()
    rng=np.random.default_rng(42);coeff=rng.dirichlet(np.ones(7),1002)
    vertices=np.vstack([np.eye(6),-np.ones((1,6))]);latents=coeff@vertices
    decoded=np.stack([np.full((128,128,3),(i+1)/10,dtype=np.float32) for i in range(7)])
    class Model:
        z_fixed=Tensor(vertices);weights=[Tensor([.5]) for _ in range(30)]
        def __call__(self,*a,**kw):return None
        def load_weights(self,path):assert Path(path).read_bytes()==b'SYNTHETIC CHECKPOINT TEST DOUBLE'
        def decode(self,*a,**kw):return Tensor(decoded),None
    model=Model();monkeypatch.setattr(model_adapter,'build_model',lambda *a:(None,model,None))
    monkeypatch.setattr(determinism,'configure_determinism',lambda *a:None)
    tf=SimpleNamespace(zeros=lambda shape:np.zeros(shape),uint8=np.uint8,
        image=SimpleNamespace(convert_image_dtype=to_uint8),
        io=SimpleNamespace(read_file=lambda name:Path(name),decode_png=lambda name,channels:Tensor(np.asarray(Image.open(name).convert('RGB')))))
    monkeypatch.setitem(sys.modules,'tensorflow',tf)
    names=['seed','python_random_seed','numpy_seed','tensorflow_seed','keras_seed','model_initialization_seed','posterior_sampling_seed','dataset_shuffle_seed']
    config=dict(K=7,latent_dimension=6,seed=113,epochs=50,batch_size=32,image_shape=[128,128,3],learning_rate=.001,
        early_stopping=False,checkpoint_rule='final epoch',smoke_test=False,status='PRIMARY RUN',optimizer='tf.keras.optimizers.Adam',
        loss_weights=constants.LOSS_WEIGHTS,protocol='archetype_stability_v1',conference_baseline_commit=constants.BASELINE_COMMIT,
        raw_unaligned_archetype_numbering=True,implementation_git_status_before_run='',implementation_git_commit=HEAD,
        determinism={**{n:113 for n in names},'deterministic_ops_enabled':True},
        process_determinism_environment={'PYTHONHASHSEED':'113','TF_DETERMINISTIC_OPS':'1'},
        manifests={part:dict(sha256=constants.MANIFEST_SHA256[part+'.csv'],full_rows=count,used_rows=count) for part,count in [('train',8011),('validation',1002),('test',1002)]})
    evidence.write_new(run/'configuration.json',config)
    losses=dict(reconstruction_sum=491.52,classification_ce=.4,kl=1.,archetype_simplex=.2,total=492.13,reconstruction_mse_per_pixel=.01,accuracy=1.)
    history=[dict(epoch=i,**{part+'_'+k:v for part in ['train','validation'] for k,v in losses.items() if k not in ['accuracy','reconstruction_mse_per_pixel']}) for i in range(1,51)]
    pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
    evidence.write_new(run/'split_metrics.json',{part:losses for part in ['train','validation','test']})
    for part in ['validation','test']:
        m=pd.read_csv(ROOT/f'research/splits/journal_v1/{part}.csv');labels=[constants.EXPECTED_CLASSES.index(x) for x in m.dx]
        m['true_label_index']=labels;m['predicted_label_index']=labels;m['predicted_dx']=m.dx;m['reconstruction_mse_per_pixel']=.01
        m.to_csv(run/(part+'_mapping.csv'),index=False);np.save(run/(part+'_coefficients.npy'),coeff);np.save(run/(part+'_latent_means.npy'),latents)
    np.save(run/'decoded_archetypes_raw.npy',decoded)
    for i in range(7):Image.fromarray(to_uint8(decoded[i]).numpy()).save(run/f'decoded_archetype_raw_A{i+1}.png')
    pd.DataFrame([row for group in nearest_real_top_n(latents,vertices,m.image_id,m.dx) for row in group]).to_csv(run/'test_nearest_real_top10_raw.csv',index=False)
    pd.DataFrame(semantic_class_rows(coeff,m.dx)).to_csv(run/'test_archetype_class_distribution_raw.csv',index=False)
    evidence.write_new(run/'test_semantic_status.json',semantic_distribution_details(coeff,m.dx))
    (run/'final.weights.h5').write_bytes(b'SYNTHETIC CHECKPOINT TEST DOUBLE')
    evidence.write_new(run/'checkpoint_reload.json',dict(loaded=True,passed=True,weight_tensor_count=30,maximum_absolute_weight_difference=0.))
    evidence.write_new(run/'warnings.json',[])
    evidence.write_new(run/'COMPLETED.json',dict(status='COMPLETED',baseline_source_sha256_before=constants.BASELINE_SOURCE_SHA256,baseline_source_sha256_after=constants.BASELINE_SOURCE_SHA256))
    def inventory():
        entries=[dict(path=p.name,bytes=p.stat().st_size,sha256=evidence.sha(p)) for p in sorted(run.iterdir()) if p.name not in ['artifact_inventory.json','COMPLETED.json']]
        if (run/'artifact_inventory.json').exists():(run/'artifact_inventory.json').unlink() # fixture only
        evidence.write_new(run/'artifact_inventory.json',{'artifacts':entries})
    inventory()
    return run,inventory


def worker_validate(run,destination,monkeypatch):
    monkeypatch.setattr(sys,'platform','linux') # fixture dispatch only; not GPU readiness
    monkeypatch.setattr(sys,'argv',['colab_gpu_worker.py','validate',str(run),'7','113',HEAD,str(destination)])
    runpy.run_path(str(ROOT/'research/tools/colab_gpu_worker.py'),run_name='__main__')


def test_complete_worker_to_actual_validator_valid_fixture(synthetic_run,tmp_path,monkeypatch):
    run,_=synthetic_run;before=evidence.hashes(run)
    with pytest.raises(AssertionError,match='reason'):evidence.validate_run(ROOT,run,7,113,[HEAD])
    worker_validate(run,tmp_path/'fixture_result.json',monkeypatch)
    result=evidence.read(tmp_path/'fixture_result.json')
    assert result['verdict']=='PASS_WITH_LIMITATIONS' and result['K']==7 and result['seed']==113
    assert result['hashes']==before==evidence.hashes(run)


@pytest.mark.parametrize('column,value',[('reason','wrong'),('status','undefined'),('coefficient_mass',np.nan),('coefficient_weighted_probability',np.inf),('enrichment_ratio',.1234)])
def test_complete_worker_rejects_semantic_mismatch(synthetic_run,tmp_path,monkeypatch,column,value):
    run,inventory=synthetic_run;p=run/'test_archetype_class_distribution_raw.csv';t=pd.read_csv(p)
    if isinstance(value,str):t[column]=t[column].astype(object)
    t.loc[0,column]=value;t.to_csv(p,index=False);inventory()
    with pytest.raises(AssertionError):worker_validate(run,tmp_path/'invalid_result.json',monkeypatch)
    assert not (tmp_path/'invalid_result.json').exists()


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_complete_worker_rejects_nonfinite_coefficients(synthetic_run,tmp_path,monkeypatch,value):
    run,inventory=synthetic_run;p=run/'test_coefficients.npy';a=np.load(p);a[0,0]=value;np.save(p,a);inventory()
    with pytest.raises(ValueError):worker_validate(run,tmp_path/'invalid_result.json',monkeypatch)


@pytest.mark.parametrize('column',['reason','status','coefficient_mass'])
def test_complete_worker_missing_required_field(synthetic_run,tmp_path,monkeypatch,column):
    run,inventory=synthetic_run;p=run/'test_archetype_class_distribution_raw.csv';t=pd.read_csv(p).drop(columns=column);t.to_csv(p,index=False);inventory()
    with pytest.raises(ValueError,match='Missing required'):worker_validate(run,tmp_path/'invalid_result.json',monkeypatch)


def test_complete_worker_zero_mass_retains_undefined(synthetic_run,tmp_path,monkeypatch):
    run,inventory=synthetic_run
    a=np.load(run/'test_coefficients.npy');a[:,0]=0;a/=a.sum(axis=1,keepdims=True)
    vertices=np.vstack([np.eye(6),-np.ones((1,6))]);z=a@vertices
    for part in ['validation','test']:
        np.save(run/(part+'_coefficients.npy'),a);np.save(run/(part+'_latent_means.npy'),z)
    m=pd.read_csv(run/'test_mapping.csv')
    pd.DataFrame([r for group in nearest_real_top_n(z,vertices,m.image_id,m.dx) for r in group]).to_csv(run/'test_nearest_real_top10_raw.csv',index=False)
    pd.DataFrame(semantic_class_rows(a,m.dx)).to_csv(run/'test_archetype_class_distribution_raw.csv',index=False)
    (run/'test_semantic_status.json').write_text(__import__('json').dumps(semantic_distribution_details(a,m.dx)))
    inventory();worker_validate(run,tmp_path/'zero_mass_result.json',monkeypatch)
    result=evidence.read(tmp_path/'zero_mass_result.json')
    assert result['verdict']=='PASS_WITH_LIMITATIONS'
    assert result['degeneration']['test']['zero_mass_count']==1


def test_D019_bound_files_match_real_receipt():
    from research.code.recovery import code_hashes
    relative='stability_workflow_local/recovered_exports/K5_seed401_20260915/RECOVERED_EXPORT.json'
    candidates=[ROOT.parent/'external'/relative,ROOT.parent.parent/relative]
    receipt=next((p for p in candidates if p.exists()),None)
    assert receipt is not None, 'Real D-019 receipt must be available'
    assert code_hashes(ROOT)==evidence.read(receipt)['code_hashes']
