"""Synthetic engineering fixtures only. No JAFFE images or training steps."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from jaffe_partb.common import ROOT, config, read, sha, write, seal, verify, approval
from jaffe_partb.data import parse_name, coverage, dataset
from jaffe_partb.analysis import coefficient_metrics, ambiguity, compute_pair, verify_pair, pair
from research.code.alignment import align_coefficients


def test_official_metadata_coverage():
    c=config(ROOT/'jaffe_partb/proposal.json')
    rows=[parse_name(n) for n in read(ROOT/'jaffe_partb/official_filename_inventory.json')['filenames']]
    assert len(rows)==213
    report=coverage(rows,c)
    assert [report[p]['images'] for p in ('train','validation','test')]==[129,41,43]
    assert all(set(v['expression_counts'])==set(c['classes']) for v in report.values())


@pytest.mark.parametrize('bad',['KM.NE1.tiff','../KM.NE1.1.tiff','YN.HA2.53.tiff','KA.XX1.1.tiff'])
def test_filename_refusal(bad):
    with pytest.raises(ValueError):parse_name(bad)


def test_leakage_refused():
    c=config(ROOT/'jaffe_partb/proposal.json');c['participants']['test'][0]='KA'
    with pytest.raises(ValueError,match='overlap'):coverage([],c)


def test_pending_approval_blocks():
    with pytest.raises(ValueError,match='pending'):
        approval(ROOT/'jaffe_partb/approval.template.json',ROOT/'jaffe_partb/proposal.json','unused',5,113)


def test_corruption_and_overwrite_refused(tmp_path):
    write(tmp_path/'a.json',dict(x=1));seal(tmp_path);verify(tmp_path)
    with pytest.raises(FileExistsError):write(tmp_path/'a.json',dict(x=2))
    (tmp_path/'a.json').write_text('{}')
    with pytest.raises(ValueError):verify(tmp_path)


def test_cluster_estimand_not_subject_average_coefficients():
    # Multiple different expressions from each person must not collapse to one vector.
    a=np.array([[1.,0],[0,1],[1,0],[0,1]])
    b=np.array([[0.,1],[1,0],[1,0],[0,1]])
    m=coefficient_metrics(a,b,['UY','UY','YM','YM'])
    assert m['image_weighted_l1']==1
    assert m['uncertainty']['metrics']['l1']['estimate']==1
    assert m['participant_metrics'][0]['l1']==2
    assert m['participants']==2


def test_undefined_not_zero_and_ambiguity():
    a=np.tile([1.,0,0],(4,1));b=np.tile([0.,1,0],(4,1))
    aligned=align_coefficients(a,b)
    m=coefficient_metrics(a,aligned.aligned_candidate,['UY','UY','YM','YM'])
    assert m['pearson_defined']==0 and all(x['pearson'] is None for x in m['columns'])
    assert ambiguity(aligned.cost_matrix,aligned.candidate_column_for_reference)['near_minimum_assignments']==2


def test_tiff_preprocessing_no_training(tmp_path):
    p=tmp_path/'KA.NE1.26.tiff';Image.fromarray(np.full((256,256),127,np.uint8)).save(p)
    rows=[dict(relative_path=p.name,expression='NE')]
    x,y=next(iter(dataset(rows,tmp_path)))
    assert x.shape==(1,128,128,3) and int(y[0])==4
    np.testing.assert_allclose(x.numpy(),127/255.,atol=1e-7)


def fixture_run(path, seed, coeff):
    path.mkdir()
    write(path/'configuration.json',dict(protocol=config(ROOT/'jaffe_partb/proposal.json'),K=5,seed=seed,config_sha256='a',manifest_sha256='b',source_head='engineering-fixture'))
    for part in ('validation','test'):
        rows=[dict(image_id=f'fixture{i}',participant_id='UY' if i<5 else 'YM',expression='NE',class_label='NE') for i in range(10)]
        write(path/f'{part}_mapping.json',rows);np.savez(path/f'{part}_outputs.npz',coefficients=coeff)
    np.save(path/'decoded_raw.npy',np.zeros((5,128,128,3),np.float32))
    write(path/'nearest_real.json',[[dict(image_id=f'fixture{i}',class_label='NE',distance=float(i),rank=i+1,archetype=a+1) for i in range(10)] for a in range(5)])
    seal(path)
    receipt=path.parent/(path.name+'_receipt.json')
    write(receipt,dict(verdict='PASS_WITH_LIMITATIONS',inventory_sha256=sha(path/'INVENTORY.json'),run=str(path.resolve()),engineering_fixture=True))
    return receipt


def test_pair_replay_and_test_does_not_select_permutation(tmp_path):
    a=np.tile([1.,0,0,0,0],(10,1));b=a[:,[1,0,2,3,4]]
    ra=fixture_run(tmp_path/'ref',113,a);rb=fixture_run(tmp_path/'cand',211,b)
    args=SimpleNamespace(reference=tmp_path/'ref',candidate=tmp_path/'cand',reference_receipt=ra,candidate_receipt=rb,output=tmp_path/'pair')
    pair(args);assert verify_pair(args.output)['verdict']=='PASS_WITH_LIMITATIONS'
    before,arr=compute_pair(args.reference,args.candidate)
    np.savez(args.candidate/'test_outputs.npz',coefficients=a)
    after,new=compute_pair(args.reference,args.candidate)
    np.testing.assert_array_equal(arr['permutation'],new['permutation'])
    assert before['coefficient']['image_weighted_l1']!=after['coefficient']['image_weighted_l1']
    with pytest.raises(ValueError):verify_pair(args.output)


@pytest.mark.parametrize('k',[3,5,7])
def test_original_model_import_and_reload_without_updates(tmp_path,k):
    import tensorflow as tf
    from research.code.model_adapter import build_model
    from jaffe_partb.training import arrays_hash
    _,m,_=build_model(ROOT,k);m(tf.zeros((1,128,128,3)),training=False)
    p=tmp_path/'final.weights.h5';m.save_weights(p)
    _,n,_=build_model(ROOT,k);n(tf.zeros((1,128,128,3)),training=False);n.load_weights(p)
    assert arrays_hash([v.numpy() for v in m.weights])==arrays_hash([v.numpy() for v in n.weights])


def test_separate_k3_configs_and_custom_reference(tmp_path):
    for name in ('ham10000','jaffe'):
        c=config(ROOT/f'jaffe_partb/configs/{name}_k3_proposed.json')
        assert c['K']==[3] and c['latent_dimension']=='K-1'
        c['seeds']=[17,29,43];c['reference_seed']=29
        p=tmp_path/(name+'.json');write(p,c)
        assert config(p)['reference_seed']==29


def test_old_experiment_id_and_implicit_seed_refused(tmp_path):
    c=config(ROOT/'jaffe_partb/proposal.json');c['experiment_id']='stability_v1';p=tmp_path/'bad.json';write(p,c)
    with pytest.raises(ValueError):config(p)
    c['experiment_id']='new_study_v1';c['seeds']=[211,307];q=tmp_path/'badref.json';write(q,c)
    with pytest.raises(ValueError):config(q)


def test_ham_new_experiment_reuses_lesion_metrics(tmp_path):
    c=config(ROOT/'jaffe_partb/configs/ham10000_k3_proposed.json')
    rng=np.random.default_rng(123)
    for name,seed in [('ref',113),('cand',211)]:
        p=tmp_path/name;p.mkdir()
        write(p/'configuration.json',dict(protocol=c,K=3,seed=seed,config_sha256='ham',manifest_sha256='hamdata',source_head='fixture'))
        for part in ('validation','test'):
            a=rng.dirichlet([1,1,1],size=10)
            np.savez(p/f'{part}_outputs.npz',coefficients=a)
            write(p/f'{part}_mapping.json',[dict(image_id=f'im{i}',lesion_id=f'lesion{i//2}',dx='nv',class_label='nv') for i in range(10)])
        np.save(p/'decoded_raw.npy',np.zeros((3,128,128,3),np.float32))
        write(p/'nearest_real.json',[[dict(image_id=f'im{i}',class_label='nv') for i in range(10)] for a in range(3)])
    result,arrays=compute_pair(tmp_path/'ref',tmp_path/'cand')
    assert result['coefficient']['unit']=='lesion_id'
    assert result['coefficient']['lesions']==5
    assert result['coefficient']['uncertainty']['unit']=='lesion_id'
    assert result['semantics']['classes']==c['classes']


def test_figures_and_configured_aggregate(tmp_path):
    from jaffe_partb.analysis import aggregate,figures
    a=np.tile([1.,0,0,0,0],(10,1))
    ra=fixture_run(tmp_path/'ref',113,a);rb=fixture_run(tmp_path/'cand',211,a)
    # Adapt only this synthetic fixture's protocol to one K/two seeds.
    for name,receipt in [('ref',ra),('cand',rb)]:
        run=tmp_path/name;value=read(run/'configuration.json');value['protocol']['K']=[5];value['protocol']['seeds']=[113,211]
        (run/'configuration.json').write_text(json.dumps(value))
        from jaffe_partb.common import inventory
        (run/'INVENTORY.json').write_text(json.dumps(inventory(run)))
        rr=read(receipt);rr['inventory_sha256']=sha(run/'INVENTORY.json');receipt.write_text(json.dumps(rr))
    args=SimpleNamespace(reference=tmp_path/'ref',candidate=tmp_path/'cand',reference_receipt=ra,candidate_receipt=rb,output=tmp_path/'pair')
    pair(args);receipt=tmp_path/'pair_receipt.json';write(receipt,verify_pair(args.output))
    figures(SimpleNamespace(pair=args.output,receipt=receipt,output=tmp_path/'figures'))
    verify(tmp_path/'figures');assert (tmp_path/'figures/pair.pdf').is_file()
    aggregate(SimpleNamespace(pair=[args.output],receipt=[receipt],output=tmp_path/'aggregate'))
    result=read(tmp_path/'aggregate/aggregate.json');assert result['statistics'][0]['sample_sd'] is None
    with pytest.raises(ValueError):
        aggregate(SimpleNamespace(pair=[args.output,args.output],receipt=[receipt,receipt],output=tmp_path/'duplicate'))
