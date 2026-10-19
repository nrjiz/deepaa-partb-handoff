"""JAFFE image metrics and participant-cluster uncertainty, explicitly not lesion metrics."""
import itertools
from pathlib import Path
import numpy as np
from research.code.alignment import align_coefficients, validate_coefficients
from research.code.stability_metrics import semantic_stability, visual_stability, compare_nearest_real, degeneration_details
from .common import plain, read, require, sha, verify, write, seal

CLASSES=['AN','DI','FE','HA','NE','SA','SU']


def coefficient_metrics(ref, cand, participants):
    ref=validate_coefficients('reference',ref);cand=validate_coefficients('candidate',cand)
    require(ref.shape==cand.shape and len(participants)==len(ref),'Paired image shapes')
    l1=np.abs(ref-cand).sum(1);agree=(ref.argmax(1)==cand.argmax(1)).astype(float)
    columns=[]
    for i in range(ref.shape[1]):
        defined=ref[:,i].std()>1e-8 and cand[:,i].std()>1e-8
        columns.append(dict(reference_column=i,pearson=float(np.corrcoef(ref[:,i],cand[:,i])[0,1]) if defined else None,
            reason=None if defined else 'degenerate_column_on_either_side',mae=float(np.abs(ref[:,i]-cand[:,i]).mean())))
    ids=sorted(set(participants));require(len(ids)>=2,'At least two test participants required')
    p=np.array(participants)
    groups=[dict(participant_id=i,images=int((p==i).sum()),l1=float(l1[p==i].mean()),agreement=float(agree[p==i].mean())) for i in ids]
    rng=np.random.default_rng(2026);sample=rng.integers(0,len(ids),size=(1000,len(ids)))
    interval={}
    for metric in ('l1','agreement'):
        values=np.array([g[metric] for g in groups]);draws=values[sample].mean(1)
        interval[metric]=dict(estimate=float(values.mean()),lower=float(np.percentile(draws,2.5)),upper=float(np.percentile(draws,97.5)))
    return dict(images=len(ref),participants=len(ids),columns=columns,pearson_defined=sum(c['pearson'] is not None for c in columns),
        image_weighted_l1=float(l1.mean()),image_weighted_agreement=float(agree.mean()),participant_metrics=groups,
        uncertainty=dict(unit='participant_id',estimand='equal-participant mean of within-participant image metrics; no averaging across expressions before metrics',
            resamples=1000,seed=2026,percentiles=[2.5,97.5],metrics=interval,
            limitation='Two participant clusters yield very limited discrete uncertainty; not evidence of population coverage or training-seed uncertainty'))


def ambiguity(cost, permutation):
    totals=np.array([sum(cost[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(len(cost)))])
    minimum=float(totals.min());bound=32*np.finfo(float).eps*max(1,abs(minimum))
    return dict(minimum_cost=minimum,saved_cost=float(sum(cost[i,j] for i,j in enumerate(permutation))),
        diagnostic_bound=bound,near_minimum_assignments=int((totals<=minimum+bound).sum()),
        test_used_for_ties=False,interpretation='Numerical ambiguity diagnostic only; saved solver permutation unchanged')


def run_receipt(run, receipt):
    run=Path(run);verify(run);r=read(receipt)
    require(r['verdict']=='PASS_WITH_LIMITATIONS' and r['inventory_sha256']==sha(run/'INVENTORY.json'),'Independent run validation required')
    require(Path(r['run']).resolve()==run.resolve(),'Receipt/run mismatch')
    return read(run/'configuration.json')


def compute_pair(reference, candidate):
    reference,candidate=Path(reference),Path(candidate)
    a,b=(read(p/'configuration.json') for p in (reference,candidate))
    c=a['protocol']
    require(c==b['protocol'] and a['K']==b['K'] and a['K'] in c['K'] and a['seed']==c['reference_seed'] and b['seed'] in c['seeds'] and b['seed']!=a['seed'],'Reference/candidate identity')
    require(a['config_sha256']==b['config_sha256'] and a['manifest_sha256']==b['manifest_sha256'] and a['source_head']==b['source_head'],'Pair experiment/source differs')
    for part in ('validation','test'):
        require(read(reference/f'{part}_mapping.json')==read(candidate/f'{part}_mapping.json'),'Pair order or membership differs')
    rv=np.load(reference/'validation_outputs.npz')['coefficients'];cv=np.load(candidate/'validation_outputs.npz')['coefficients']
    match=align_coefficients(rv,cv);perm=match.candidate_column_for_reference
    rt=np.load(reference/'test_outputs.npz')['coefficients'];ct=np.load(candidate/'test_outputs.npz')['coefficients'];aligned=ct[:,perm]
    mapping=read(reference/'test_mapping.json');labels=[r['class_label'] for r in mapping]
    semantics=semantic_stability(rt,labels,aligned,labels,classes=c['classes'])
    # Existing low-level nearest comparison uses an internal key named dx; this
    # explicit, temporary translation is expression-only and never creates lesion IDs.
    def nearest(p):
        return [[dict(x,dx=x['class_label']) for x in group] for group in read(p/'nearest_real.json')]
    nearest_rows=compare_nearest_real(nearest(reference),nearest(candidate),perm,classes=c['classes'])
    for row in nearest_rows:row['label_js_base2']=row.pop('diagnosis_jensen_shannon_divergence_base2')
    if c['dataset']=='JAFFE':
        coefficient=coefficient_metrics(rt,aligned,[r['participant_id'] for r in mapping])
        coefficient['unit']='image_with_participant_cluster_uncertainty'
    else:
        import pandas as pd
        from research.code.stability_metrics import coefficient_stability, paired_lesion_coefficients, lesion_bootstrap_coefficient_stability
        frame=pd.DataFrame(mapping)
        coefficient=coefficient_stability(rt,frame,aligned,frame)
        ref_lesion,cand_lesion,ids=paired_lesion_coefficients(rt,frame,aligned,frame)
        coefficient['lesion_mapping']=ids.to_dict('records')
        coefficient['uncertainty']=lesion_bootstrap_coefficient_stability(ref_lesion,cand_lesion)
        coefficient['unit']='lesion_id'
    summary=dict(protocol=c,config_sha256=a['config_sha256'],manifest_sha256=a['manifest_sha256'],source_head=a['source_head'],K=a['K'],reference_seed=c['reference_seed'],candidate_seed=b['seed'],matching_partition='validation',test_used_for_matching=False,
        assignment=match.summary(),ambiguity=ambiguity(match.cost_matrix,perm),coefficient=coefficient,
        semantics=semantics,nearest_real=nearest_rows,visual=visual_stability(np.load(reference/'decoded_raw.npy'),np.load(candidate/'decoded_raw.npy'),perm),
        degeneration=dict(reference=degeneration_details(rt),candidate_raw=degeneration_details(ct)),
        limitations=['Ambiguous matching is not uniquely identified semantic correspondence','Undefined correlations and zero-mass semantics retained']+(['Requested-expression weighted profiles, not diagnosis or internal emotion'] if c['dataset']=='JAFFE' else ['New HAM experiment; never pool with accepted stability_v1 results']))
    arrays=dict(cost_matrix=match.cost_matrix,cost_method_mask=match.cost_method_matrix.astype(str),permutation=perm,
        reference_validation=rv,candidate_validation_raw=cv,candidate_validation_aligned=cv[:,perm],
        reference_test=rt,candidate_test_raw=ct,candidate_test_aligned=aligned)
    return plain(summary),arrays


def pair(args):
    run_receipt(args.reference,args.reference_receipt);run_receipt(args.candidate,args.candidate_receipt)
    summary,arrays=compute_pair(args.reference,args.candidate)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'summary.json',summary);np.savez_compressed(out/'alignment_and_outputs.npz',**arrays)
    write(out/'provenance.json',dict(reference=str(Path(args.reference).resolve()),candidate=str(Path(args.candidate).resolve()),
        reference_receipt=str(Path(args.reference_receipt).resolve()),candidate_receipt=str(Path(args.candidate_receipt).resolve()),
        reference_receipt_sha256=sha(args.reference_receipt),candidate_receipt_sha256=sha(args.candidate_receipt)))
    seal(out)


def verify_pair(path):
    path=Path(path);verify(path);p=read(path/'provenance.json')
    for side in ('reference','candidate'):
        require(sha(p[side+'_receipt'])==p[side+'_receipt_sha256'],'Validation receipt changed')
        run_receipt(p[side],p[side+'_receipt'])
    expected,arrays=compute_pair(p['reference'],p['candidate'])
    require(expected==read(path/'summary.json'),'Independent pair recomputation differs')
    saved=np.load(path/'alignment_and_outputs.npz',allow_pickle=False)
    require(set(saved.files)==set(arrays),'Pair arrays missing')
    for n,a in arrays.items():np.testing.assert_array_equal(a,saved[n])
    return dict(verdict='PASS_WITH_LIMITATIONS',pair=str(path.resolve()),inventory_sha256=sha(path/'INVENTORY.json'),
        matching_partition='validation',test_used_for_matching=False,undefined_values_preserved=True,ambiguity=expected['ambiguity'])


def aggregate(args):
    require(len(args.pair)==len(args.receipt) and len(args.pair)>0,'Pair/receipt count mismatch')
    summaries=[];identities=[]
    for p,r in zip(args.pair,args.receipt):
        verify(p);rr=read(r);require(rr['inventory_sha256']==sha(Path(p)/'INVENTORY.json') and rr['verdict']=='PASS_WITH_LIMITATIONS','Pair receipt mismatch')
        require(Path(rr['pair']).resolve()==Path(p).resolve(),'Pair receipt path')
        s=read(Path(p)/'summary.json');summaries.append(s);identities.append((s['K'],s['candidate_seed']))
    c=summaries[0]['protocol'];expected=set(itertools.product(c['K'],[s for s in c['seeds'] if s!=c['reference_seed']]))
    require(len(identities)==len(expected) and set(identities)==expected,'Configured pair matrix incomplete or duplicates')
    require(all(all(s[k]==summaries[0][k] for k in ('protocol','config_sha256','manifest_sha256','source_head')) for s in summaries),'Mixed experiment/configuration/source')
    stats=[]
    for k in c['K']:
        for metric in (('image_weighted_l1','image_weighted_agreement') if c['dataset']=='JAFFE' else ('coefficient_vector_l1_mean','dominant_archetype_agreement')):
            values=[s['coefficient'][metric] for s in summaries if s['K']==k]
            stats.append(dict(K=k,metric=metric,raw=values,defined_pairs=len(values),mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,sample_sd_reason=None if len(values)>1 else 'fewer_than_two_pairs',minimum=min(values),maximum=max(values),seed_confidence_interval=None))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'aggregate.json',dict(protocol=c,pairs=summaries,statistics=stats,limitations=['Shared reference pairs are dependent','No seed-level inferential interval','Preserve all collapsed seeds and null values']))
    write(out/'provenance.json',[dict(pair=str(Path(p).resolve()),pair_inventory_sha256=sha(Path(p)/'INVENTORY.json'),receipt_sha256=sha(r)) for p,r in zip(args.pair,args.receipt)])
    seal(out)


def figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(args.pair);verify(p);r=read(args.receipt)
    require(r['inventory_sha256']==sha(p/'INVENTORY.json') and r['verdict']=='PASS_WITH_LIMITATIONS','Verified pair required')
    s=read(p/'summary.json');a=np.load(p/'alignment_and_outputs.npz');out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    im=ax[0].imshow(a['cost_matrix'],vmin=0,vmax=2);ax[0].scatter(a['permutation'],np.arange(s['K']),facecolors='none',edgecolors='red',s=120)
    ax[0].set(xlabel='Candidate raw index (zero-based)',ylabel='Reference raw index (zero-based)',title='Validation-only assignment')
    fig.colorbar(im,ax=ax[0])
    for i,key in enumerate(('reference_test','candidate_test_raw')):
        ax[1].plot(np.arange(s['K'])+1,a[key].mean(0),'o-',label=key)
    ax[1].set(xlabel='Raw archetype index (not matched)',ylabel='Mean test coefficient',title='Usage; distinct images do not prove meaning');ax[1].legend()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'pair.{ext}',dpi=300)
    plt.close(fig);write(out/'provenance.json',dict(pair=str(p.resolve()),inventory_sha256=sha(p/'INVENTORY.json'),caption='Saved validation costs and unchanged permutation; raw coefficient usage is not semantic identity. Ambiguity and undefined metrics are in the pair summary. No face images included.'))
    seal(out)
