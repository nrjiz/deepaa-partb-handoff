"""Fresh-process GPU training/benchmark and unchanged artifact verification.

Training settings are imported from frozen code. This file is not a Windows
training entry point. Benchmarks read frozen TRAIN membership only.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.code.workflow_evidence import require, read, write_new

PINS={'tensorflow':'2.21.0','keras':'3.15.1','numpy':'2.5.2','pandas':'3.0.5',
      'scipy':'1.18.1','h5py':'3.14.0','pillow':'12.3.0','matplotlib':'3.11.1','pytest':'9.1.1'}


def gpu_environment():
    require(sys.platform.startswith('linux'),'GPU entry point is Linux-only; Windows training prohibited')
    require(platform.python_version()=='3.13.3','Use isolated Python 3.13.3; do not accept Colab defaults')
    for name,version in PINS.items():
        require(importlib.metadata.version(name)==version,f'Environment mismatch: {name}')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    import tensorflow as tf
    require(tf.keras.mixed_precision.global_policy().name=='float32','Float32 policy required')
    devices=tf.config.list_physical_devices('GPU')
    require(devices,'No TensorFlow-visible GPU; CPU training forbidden')
    for device in devices: tf.config.experimental.set_memory_growth(device,True)
    tf.config.experimental.enable_op_determinism()
    with tf.device('/GPU:0'):
        a=tf.ones((32,32),tf.float32);b=tf.linalg.matmul(a,a)
    require('GPU' in b.device,'GPU placement failed')
    b.numpy()
    return dict(python=platform.python_version(),executable=sys.executable,
                versions={n:importlib.metadata.version(n) for n in PINS},
                freeze=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True).splitlines(),
                devices=[dict(name=d.name,details={k:str(v) for k,v in tf.config.experimental.get_device_details(d).items()}) for d in devices],
                tensorflow_build={k:str(v) for k,v in tf.sysconfig.get_build_info().items()},
                nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),float_policy='float32',
                deterministic_ops=True,gpu_test_device=b.device)


def benchmark(k,seed,dataset,destination):
    environment=gpu_environment()
    import numpy as np
    import tensorflow as tf
    from research.code.data import load_manifest,make_tf_dataset
    from research.code.constants import MANIFEST_SHA256
    from research.code.determinism import configure_determinism
    from research.code.model_adapter import build_model
    d=configure_determinism(seed)
    require(d.deterministic_ops_enabled and not d.warnings,'Determinism unsupported')
    frame=load_manifest(ROOT/'research/splits/journal_v1/train.csv',
        [dataset/f'HAM10000_images_part_{i}' for i in (1,2)],MANIFEST_SHA256['train.csv'],8011)
    # Exactly one training-only batch, repeated solely for operational timing.
    batch=next(iter(make_tf_dataset(frame.iloc[:32].copy(),128,32,shuffle=False,seed=None)))
    times=[];losses=[]
    with tf.device('/GPU:0'):
        baseline,model,cfg=build_model(ROOT,k)
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001)
        for step in range(23):
            started=time.perf_counter()
            values=baseline.train_step(model,optimizer,*batch,cfg)
            vals=[float(x.numpy()) for x in values]
            require(np.isfinite(vals).all(),'Nonfinite smoke values')
            losses.append(vals)
            if step>=3: times.append(time.perf_counter()-started)
    require(any('GPU' in str(getattr(w,'value',w).device) for w in model.weights),'Model variables not on GPU')
    digest=hashlib.sha256()
    for w in model.weights: digest.update(w.numpy().tobytes())
    digest.update(np.asarray(losses).tobytes())
    write_new(destination,dict(status='NON_SCIENTIFIC_TRAINING_ONLY_BENCHMARK',K=k,seed=seed,
        rows=32,warmup_steps=3,timed_steps=20,median_seconds=statistics.median(times),
        training_loop_estimate_seconds=statistics.median(times)*251*50,
        estimate_excludes='validation, full dataset I/O, compilation, final inference/export, persistence and interruptions',
        gpu_memory=tf.config.experimental.get_memory_info('GPU:0'),fingerprint=digest.hexdigest(),
        environment=environment,determinism=d.to_dict(),scientific_evidence=False))


def main():
    mode,*a=sys.argv[1:]
    require(sys.platform.startswith('linux'),'Colab workers are Linux-only')
    if mode=='environment':
        write_new(Path(a[0]),gpu_environment())
    elif mode=='data':
        from research.code.data import load_journal_v1
        from research.code.workflow_evidence import sha
        dataset=Path(a[0]);split=ROOT/'research/splits/journal_v1'
        frames=load_journal_v1(split/'train.csv',split/'validation.csv',split/'test.csv',
                              [dataset/f'HAM10000_images_part_{i}' for i in (1,2)])
        write_new(Path(a[1]),dict(status='VERIFIED_FROZEN_MEMBERSHIP_ORDER_AND_DISJOINTNESS',
            partitions={n:dict(images=len(f),lesions=int(f.lesion_id.nunique()),manifest_sha256=sha(split/(n+'.csv')))
                        for n,f in frames.items()},jpeg_content_verification='parent verified all source-inventory hashes'))
    elif mode=='benchmark':
        benchmark(int(a[0]),int(a[1]),Path(a[2]),Path(a[3]))
    elif mode=='train':
        gpu_environment()
        from research.code.runner import RunSpec,execute_run
        k,seed=int(a[0]),int(a[1]);dataset=Path(a[3])
        require((k,seed) in [(5,503),(7,113),(7,211),(7,307),(7,401),(7,503)],'D-020 training scope')
        require(os.environ.get('PYTHONHASHSEED')==str(seed) and os.environ.get('TF_DETERMINISTIC_OPS')=='1','Process seed')
        split=ROOT/'research/splits/journal_v1'
        execute_run(RunSpec(repository=ROOT,k=k,seed=seed,output_dir=Path(a[2]),
            train_manifest=split/'train.csv',validation_manifest=split/'validation.csv',test_manifest=split/'test.csv',
            image_dirs=[dataset/f'HAM10000_images_part_{i}' for i in (1,2)]))
    elif mode=='validate':
        from research.code.workflow_evidence import validate_run
        from research.code.workflow import HISTORICAL
        from research.code.recovery import TRAINING_REVISION
        run=Path(a[0]);k,s=int(a[1]),int(a[2])
        if (k,s)==(5,401):
            from research.code.recovery import registered_path
            state=ROOT.parent/'external/stability_workflow_local'
            require(registered_path(ROOT,state,k,s,ROOT/'research/runs/stability_v1/K5/seed_401')==run.resolve(),
                    'D-019 registered path/receipt mismatch')
        approved=list(HISTORICAL.values())+[TRAINING_REVISION,a[3]]
        write_new(Path(a[4]),validate_run(ROOT,run,k,s,approved))
    elif mode=='compare':
        ref,cand,pair=map(Path,a)
        for tool,extra in [('align_validation_coefficients.py',['--output-dir',str(pair/'alignment')]),
                           ('evaluate_stability_pair.py',['--alignment-dir',str(pair/'alignment'),'--output-dir',str(pair/'test')])]:
            subprocess.run([sys.executable,'-B',str(ROOT/'research/tools'/tool),'--reference-run',str(ref),
                            '--candidate-run',str(cand),*extra],check=True)
    elif mode=='pair_verify':
        from research.code.workflow_evidence import verify_pair
        pair,ref,cand=map(Path,a[:3])
        write_new(Path(a[5]),verify_pair(ROOT,pair,ref,cand,int(a[3]),int(a[4])))
    elif mode=='aggregate':
        from research.code.workflow_evidence import aggregate
        inputs=read(Path(a[0]));aggregate(Path(a[1]),inputs['runs'],inputs['pairs'])
    else:
        raise ValueError('Unknown worker action')


if __name__=='__main__':
    main()
