"""One explicitly approved JAFFE run. Reuses the protected original DeepAA model."""
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .common import ROOT, approval, config, external, git, inventory, lock, read, require, seal, sha, write
from .data import dataset, load


def evaluate(base, model, cfg, rows, data, c):
    from research.code.runner import _evaluate_losses, _collect_outputs
    ds=dataset(rows,data,c=c)
    losses=_evaluate_losses(base,model,cfg,ds)
    outputs=_collect_outputs(model,ds)
    losses['accuracy']=float(np.mean(outputs['labels']==outputs['predictions']))
    losses['reconstruction_mse_per_pixel']=float(outputs['reconstruction_mse_per_pixel'].mean())
    return losses,outputs


def arrays_hash(values):
    import hashlib
    return [dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for a in values]


def train(args):
    c=config(args.config)
    require(args.k in c['K'] and args.seed in c['seeds'], 'Unregistered K/seed')
    require(os.environ.get('PYTHONHASHSEED')==str(args.seed) and os.environ.get('TF_DETERMINISTIC_OPS')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1', 'Fresh deterministic CPU process required')
    a=approval(args.approval,args.config,args.manifest,args.k,args.seed)
    from research.code.constants import BASELINE_SOURCE,BASELINE_SOURCE_SHA256
    require(sha(ROOT/BASELINE_SOURCE)==BASELINE_SOURCE_SHA256,'Protected model byte identity differs')
    frames=load(args.manifest,args.data,c,sha(args.config))
    root=external(args.workspace)/c['experiment_id'];root.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(root).free>=2*1024**3, 'At least 2 GiB workspace disk headroom required')
    run=root/'runs'/f'K{args.k}_seed{args.seed}'
    claims=root/'claims';claims.mkdir(exist_ok=True)
    with lock(root):
        identity=dict(experiment_id=c['experiment_id'],config_sha256=sha(args.config),manifest_sha256=sha(args.manifest),source_head=git('rev-parse','HEAD'))
        if (root/'EXPERIMENT.json').exists():
            require(read(root/'EXPERIMENT.json')==identity,'Experiment identifier already binds different settings/data/source')
        else:
            write(root/'EXPERIMENT.json',identity)
        require(not run.exists(), 'Output already exists; no automatic replacement')
        write(claims/f'K{args.k}_seed{args.seed}.json',dict(pid=os.getpid(),source_head=git('rev-parse','HEAD'),approval_sha256=sha(args.approval),output=str(run),utc=datetime.now(timezone.utc).isoformat(),no_retry=True))
        run.mkdir(parents=True,exist_ok=False)
        try:
            from research.code.determinism import configure_determinism
            det=configure_determinism(args.seed)
            require(det.deterministic_ops_enabled and not det.warnings,'Determinism requirements failed')
            import tensorflow as tf
            from research.code.model_adapter import build_model, provenance
            from research.code.runner import _mean_components
            from research.code.stability_metrics import degeneration_details, nearest_real_top_n
            require(not tf.config.list_physical_devices('GPU'), 'This reviewed implementation is CPU only')
            base,model,cfg=build_model(ROOT,args.k)
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001)
            source_head=git('rev-parse','HEAD')
            write(run/'configuration.json',dict(protocol=c,K=args.k,seed=args.seed,latent_dimension=args.k-1,epochs=50,
                  source_head=source_head,model_provenance=provenance(),config_sha256=sha(args.config),manifest_sha256=sha(args.manifest),
                  approval=a,approval_sha256=sha(args.approval),command=sys.argv,initialization='fresh',sampling='original_stochastic',resume=False))
            shutil.copyfile(args.manifest,run/'DATASET_MANIFEST.json')
            write(run/'environment.json',dict(python=sys.version,platform=sys.platform,devices=[str(x) for x in tf.config.list_physical_devices()],
                packages={n:importlib.metadata.version(n) for n in ('tensorflow','keras','numpy','pandas','pillow','scipy','matplotlib','h5py')},
                freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),determinism=det.to_dict()))
            history=[];ds=dataset(frames['train'],args.data,args.seed,c);start=time.monotonic()
            with (run/'training.log').open('x',encoding='utf-8') as log:
                for epoch in range(1,51):
                    losses=[];t=time.monotonic()
                    for batch,(x,y) in enumerate(ds,start=1):
                        losses.append([float(v.numpy()) for v in base.train_step(model,optimizer,x,y,cfg)])
                        event=dict(epoch=epoch,epochs=50,batch=batch,batches=math.ceil(len(frames['train'])/32),utc=datetime.now(timezone.utc).isoformat())
                        print(event,flush=True);log.write(str(event)+'\n');log.flush()
                    from research.code.runner import _evaluate_losses
                    val=_evaluate_losses(base,model,cfg,dataset(frames['validation'],args.data,c=c))
                    history.append(dict(epoch=epoch,train=_mean_components(losses),validation=val,seconds=time.monotonic()-t))
                    write(run/f'epoch_{epoch:02d}.json',history[-1])
            write(run/'history.json',history)
            model.save_weights(run/'final.weights.h5')
            write(run/'weight_tensors.json',arrays_hash([x.numpy() for x in model.weights]))
            metrics={};deg={}
            for part,rows in frames.items():
                metrics[part],out=evaluate(base,model,cfg,rows,args.data,c)
                np.savez_compressed(run/f'{part}_outputs.npz',**out)
                write(run/f'{part}_mapping.json',rows)
                deg[part]=degeneration_details(out['coefficients'])
            write(run/'metrics.json',metrics);write(run/'degeneration.json',deg)
            decoded=model.decode(tf.constant(model.z_fixed.numpy(),tf.float32),training=False)[0].numpy()
            np.save(run/'decoded_raw.npy',decoded,allow_pickle=False)
            np.save(run/'z_fixed.npy',model.z_fixed.numpy(),allow_pickle=False)
            for i,img in enumerate(decoded,1):
                tf.io.write_file(str(run/f'decoded_raw_A{i}.png'),tf.io.encode_png(tf.image.convert_image_dtype(img,tf.uint8,saturate=True)))
            test=np.load(run/'test_outputs.npz');rows=frames['test']
            nearest=nearest_real_top_n(test['latent_means'],model.z_fixed.numpy(),[r['image_id'] for r in rows],[r['class_label'] for r in rows])
            for group in nearest:
                for row in group:row['class_label']=row.pop('dx')
            write(run/'nearest_real.json',nearest)
            require(git('rev-parse','HEAD')==source_head and not git('status','--porcelain','--untracked-files=all'), 'Source changed during execution')
            write(run/'COMPLETED.json',dict(status='COMPLETED_NOT_YET_INDEPENDENTLY_VALIDATED',epochs=50,updates=int(optimizer.iterations.numpy()),seconds=time.monotonic()-start))
            seal(run)
        except BaseException as exc:
            write(run/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False))
            raise
    print('Completed export; run independent validate command:',run,flush=True)


def validate_run(run, data, config_path):
    """Fresh-process reload, numerical export replay and hash checks; no optimizer updates."""
    import tensorflow as tf
    from research.code.model_adapter import build_model, provenance
    from research.code.stability_metrics import degeneration_details, nearest_real_top_n
    from .common import verify
    run=Path(run);verify(run);c=config(config_path);record=read(run/'configuration.json')
    require(not (run/'FAILED.json').exists(), 'Failed run cannot validate as complete')
    require(record['protocol']==c and record['config_sha256']==sha(config_path),'Configuration differs')
    require(record['model_provenance']==provenance() and record['sampling']=='original_stochastic' and record['initialization']=='fresh','Model provenance')
    require(record['K'] in c['K'] and record['seed'] in c['seeds'] and record['latent_dimension']==record['K']-1 and record['epochs']==50,'Run scope')
    require(sha(run/'DATASET_MANIFEST.json')==record['manifest_sha256'],'Manifest differs')
    frames=load(run/'DATASET_MANIFEST.json',data,c,sha(config_path))
    hist=read(run/'history.json');require([r['epoch'] for r in hist]==list(range(1,51)),'50-epoch continuity')
    for h in hist:
        require(read(run/f"epoch_{h['epoch']:02d}.json")==h,'Epoch record differs')
        require(all(np.isfinite(v) for phase in ('train','validation') for v in h[phase].values()),'Nonfinite history')
    done=read(run/'COMPLETED.json');require(done['epochs']==50 and done['updates']==50*math.ceil(len(frames['train'])/32),'Update count')
    base,model,cfg=build_model(ROOT,record['K']);model(tf.zeros((1,128,128,3)),training=False)
    model.load_weights(run/'final.weights.h5')
    require(arrays_hash([x.numpy() for x in model.weights])==read(run/'weight_tensors.json'),'Saved tensor reload differs')
    metrics=read(run/'metrics.json');deg=read(run/'degeneration.json')
    for part,rows in frames.items():
        require(read(run/f'{part}_mapping.json')==rows,'Mapping differs')
        measured,out=evaluate(base,model,cfg,rows,data,c)
        saved=np.load(run/f'{part}_outputs.npz',allow_pickle=False)
        require(set(saved.files)==set(out),'Export keys')
        for key in out:np.testing.assert_allclose(out[key],saved[key],rtol=1e-6,atol=1e-7)
        for key in measured:np.testing.assert_allclose(measured[key],metrics[part][key],rtol=1e-6,atol=1e-7)
        require(degeneration_details(saved['coefficients'])==deg[part],'Degeneration report differs')
    decoded=model.decode(tf.constant(model.z_fixed.numpy(),tf.float32),training=False)[0].numpy()
    np.testing.assert_array_equal(model.z_fixed.numpy(),np.load(run/'z_fixed.npy'))
    np.testing.assert_allclose(decoded,np.load(run/'decoded_raw.npy'),rtol=1e-6,atol=1e-7)
    from PIL import Image
    for i,img in enumerate(np.load(run/'decoded_raw.npy'),1):
        expected=tf.image.convert_image_dtype(img,tf.uint8,saturate=True).numpy()
        np.testing.assert_array_equal(np.asarray(Image.open(run/f'decoded_raw_A{i}.png')),expected)
    rows=frames['test'];test=np.load(run/'test_outputs.npz')
    nearest=nearest_real_top_n(test['latent_means'],np.load(run/'z_fixed.npy'),[r['image_id'] for r in rows],[r['class_label'] for r in rows])
    for group in nearest:
        for item in group:item['class_label']=item.pop('dx')
    require(nearest==read(run/'nearest_real.json'),'Nearest-real report differs')
    return dict(verdict='PASS_WITH_LIMITATIONS',run=str(run.resolve()),inventory_sha256=sha(run/'INVENTORY.json'),source_head=record['source_head'],
        validator_head=git('rev-parse','HEAD'),epochs=50,weight_tensors_exact=True,replay_tolerance=dict(rtol=1e-6,atol=1e-7),
        limitations=['New-experiment validator; not a retrospective accepted HAM10000 receipt','Artifact validation does not establish meaningful archetypes']+(['Two held-out participants; no population-generalization claim','Requested expressions are not internal emotion ground truth'] if c['dataset']=='JAFFE' else ['Inherited lesion bootstrap conditions on fixed trained models; not training-seed uncertainty']))
