"""D-019 fixtures: no TensorFlow training or optimizer steps."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.code.stability_metrics import (
    coefficient_weighted_class_distribution, semantic_distribution_details,
    semantic_class_rows, semantic_stability, semantic_summary, semantic_comparison_rows,
    coefficient_stability_from_lesions, StabilityInputError,
)
from research.code.alignment import align_coefficients
from research.code.workflow_evidence import write_new, hashes, sha, aggregate
from research.code import recovery


def test_zero_mass_nulls_and_positive_tiny_mass_unchanged():
    a = np.array([[0,1e-100,1],[0,2e-100,1]],dtype=float)
    details = semantic_distribution_details(a,['nv','mel'])
    assert details[0] == dict(raw_archetype=1,coefficient_mass=0.,status='undefined',
                             reason='zero_coefficient_mass',probabilities=None)
    assert details[1]['status']=='defined' and details[1]['coefficient_mass']>0
    probs = coefficient_weighted_class_distribution(a,['nv','mel'])
    np.testing.assert_allclose(sorted(probs[1])[-2:],[1/3,2/3])
    json.dumps(details,allow_nan=False)
    rows = semantic_class_rows(a,['nv','mel'])
    assert all(r['coefficient_weighted_probability'] is None and r['enrichment_ratio'] is None for r in rows[:7])


@pytest.mark.parametrize('side',['reference','candidate','both'])
def test_either_side_zero_means_undefined_js(side):
    good=np.array([[.2,.8],[.8,.2]])
    bad=np.array([[0.,1.],[0.,1.]])
    ref=bad if side in ['reference','both'] else good
    cand=bad if side in ['candidate','both'] else good
    result=semantic_stability(ref,['nv','mel'],cand,['nv','mel'])
    assert result['jensen_shannon_divergence_base2'][0] is None
    assert result['defined_pairs']==1 and result['total_pairs']==2
    assert result['mean_scope']=='defined pairs only'
    assert result['mean_jensen_shannon_divergence_base2']==result['jensen_shannon_divergence_base2'][1]
    json.dumps(semantic_summary(result),allow_nan=False)
    assert semantic_comparison_rows(result)[0]['reason']=='zero_mass_on_either_side'


def test_no_valid_matches_mean_is_null():
    a=np.array([[0.,1.],[0.,1.]])
    s=semantic_stability(a,['nv','mel'],a[:,::-1],['nv','mel'])
    assert s['defined_pairs']==0 and s['mean_jensen_shannon_divergence_base2'] is None


@pytest.mark.parametrize('bad',[np.array([[-1.,2.],[0.,1.]]),np.array([[np.nan,1.],[0.,1.]])])
def test_corrupt_coefficients_still_rejected(bad):
    with pytest.raises(StabilityInputError):semantic_distribution_details(bad,['nv','mel'])


def test_matching_unchanged_and_correlations_null():
    ref=np.random.default_rng(113).dirichlet(np.ones(5),size=40)
    cand=np.zeros_like(ref);cand[:,4]=1
    alignment=align_coefficients(ref,cand)
    assert alignment.summary()['fallback_cost_cells']==25
    for i in range(5):
        for j in range(5):assert alignment.cost_matrix[i,j]==2*np.abs(ref[:,i]-cand[:,j]).mean()
    metrics=coefficient_stability_from_lesions(ref,alignment.aligned_candidate)
    assert all(r['pearson_correlation'] is None and r['correlation_defined'] is False for r in metrics['per_archetype'])
    assert np.isfinite(metrics['coefficient_vector_l1_mean'])


def receipt_fixture(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    state=tmp_path/'state';state.mkdir()
    original=root/'research/runs/stability_v1/K5/seed_401';original.mkdir(parents=True)
    write_new(original/'FAILED.json',{'status':'FAILED'})
    recovered=state/'recovered';recovered.mkdir()
    (recovered/'FAILED.json').write_bytes((original/'FAILED.json').read_bytes())
    write_new(recovered/'recovery_validation.json',dict(verdict='PASS_WITH_LIMITATIONS',K=5,seed=401))
    write_new(recovered/'inference_equivalence.json',dict(passed=True,optimizer_steps=0,
        checkpoint_sha256=recovery.CHECKPOINT,decoded_max_absolute_difference=0,
        splits={part:dict(rows=1002,coefficients_exact=True,latent_means_exact=True,predictions_exact=True,
                         reconstruction_mse_equivalent=True) for part in ['validation','test']}))
    monkeypatch.setattr(recovery,'code_hashes',lambda root:{'fixture':'code'})
    def verify_original(root,path,expected):
        assert path==original
        if hashes(path)!=expected:raise ValueError('Original evidence changed')
    monkeypatch.setattr(recovery,'verify_original',verify_original)
    write_new(recovered/recovery.RECEIPT,dict(schema='seed401_export_recovery_v1',decision='D-019',
        status='VERIFIED_RECOVERED_EXPORT',optimizer_steps=0,training_revision=recovery.TRAINING_REVISION,
        checkpoint_sha256=recovery.CHECKPOINT,code_hashes={'fixture':'code'},original_run=str(original),
        original_hashes=hashes(original),recovered_hashes=hashes(recovered),external_evidence=[]))
    folder=state/'recoveries';folder.mkdir()
    write_new(folder/'K5_seed401.json',dict(recovered_run=str(recovered),receipt_sha256=sha(recovered/recovery.RECEIPT)))
    return root,state,original,recovered


def test_registered_recovery_identity_and_hashes(tmp_path,monkeypatch):
    root,state,original,recovered=receipt_fixture(tmp_path,monkeypatch)
    assert recovery.registered_path(root,state,5,401,original)==recovered
    assert recovery.registered_path(root,state,5,503,root/'missing')==root/'missing'
    assert (original/'FAILED.json').exists() and not (original/'COMPLETED.json').exists()


@pytest.mark.parametrize('corruption',['missing','tampered','original','receipt','completion','code'])
def test_invalid_recovery_rejected(tmp_path,monkeypatch,corruption):
    root,state,original,recovered=receipt_fixture(tmp_path,monkeypatch)
    if corruption=='missing':(recovered/'inference_equivalence.json').unlink()
    if corruption=='tampered':(recovered/'recovery_validation.json').write_text('{}')
    if corruption=='original':(original/'FAILED.json').write_text('{}')
    if corruption=='receipt':(recovered/recovery.RECEIPT).write_text('{}')
    if corruption=='completion':write_new(recovered/'COMPLETED.json',{})
    if corruption=='code':monkeypatch.setattr(recovery,'code_hashes',lambda root:{'fixture':'changed'})
    with pytest.raises((ValueError,FileNotFoundError)):
        recovery.registered_path(root,state,5,401,original)


def test_summary_null_csv_and_denominators(tmp_path):
    from research.tests.test_workflow import fixture_summary_inputs
    runs,pairs=fixture_summary_inputs(tmp_path,[(5,113),(5,211),(5,307)])
    path=Path(pairs[0]['pair'])/'test/semantic_stability.csv'
    frame=pd.read_csv(path);frame.loc[0,'jensen_shannon_divergence_base2']=np.nan
    frame.to_csv(path,index=False)
    pairs[0]['hashes']=hashes(Path(pairs[0]['pair']))
    pairs[0]['summary']['semantic'].update(defined_pairs=4,total_pairs=5,undefined_pairs=1,mean_scope='defined pairs only')
    result=aggregate(tmp_path/'summary',runs,pairs)
    row=next(r for r in result['descriptive_statistics'] if r['metric']=='semantic_js_base2' and r['archetype']==1)
    assert row['defined_values']==1 and row['undefined_values']==1 and row['sample_standard_deviation'] is None
    assert len(result['registered_matrix'])==10
    assert result['degeneration_counts']['5']['undefined_semantic_matches']==1
    raw=pd.read_csv(tmp_path/'summary/raw_metrics.csv')
    assert raw[(raw.metric=='semantic_js_base2')&(raw.archetype==1)&(raw.seed==211)].value.isna().all()


def test_recovered_seed_is_validated_not_retrained(tmp_path,monkeypatch):
    from research.code.workflow import Workflow
    from research.code import workflow as module
    w=Workflow(tmp_path/'repo',tmp_path/'state','a'*40)
    w.session=w.state/'session';w.session.mkdir(parents=True)
    paths={}
    for seed in [113,211,307,401]:
        paths[seed]=tmp_path/f'seed{seed}';paths[seed].mkdir()
        write_new(paths[seed]/('RECOVERED_EXPORT.json' if seed==401 else 'COMPLETED.json'),{})
    monkeypatch.setattr(w,'run_path',lambda k,s:paths.get(s,tmp_path/f'absent{k}_{s}'))
    seen=[]
    monkeypatch.setattr(w,'validate',lambda k,s:(seen.append(('validate',s)) or dict(K=k,seed=s)))
    monkeypatch.setattr(w,'backup_run',lambda report:None)
    monkeypatch.setattr(w,'pair',lambda *a,**kw:None)
    def launch(k,s):
        seen.append(('launch',s));raise RuntimeError('stub boundary; never train')
    monkeypatch.setattr(w,'launch',launch)
    with pytest.raises(RuntimeError,match='stub boundary'):w.work(True)
    assert seen==[('validate',113),('validate',211),('validate',307),('validate',401),('launch',503)]
