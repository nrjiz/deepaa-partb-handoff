"""Bounded Route B engineering only. Never creates a production approval/claim."""
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from research.code.workflow_evidence import read,write_new,sha,require
from research.code.colab_transfer import inventory
PURPOSE='BOUNDED_RECOVERY_ENGINEERING'
SCOPE=dict(K=7,latent_dimension=6,seed=113,mode='bounded recovery engineering',owner='Colab')

def snapshot(source,dest):
    require(not dest.exists(),'Snapshot destination already exists')
    require(not any(p.is_symlink() for p in source.rglob('*')),'Snapshot symlink refused')
    before=inventory(source);dest.mkdir(parents=True)
    for name,item in before.items():
        src=source/name;target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,target.open('xb') as b:
            shutil.copyfileobj(a,b);b.flush();os.fsync(b.fileno())
        require(sha(target)==item['sha256'],'Snapshot readback mismatch')
    require(inventory(source)==before and inventory(dest)==before,'Snapshot inventory differs')
    write_new(dest/'SNAPSHOT_COMPLETE.json',dict(purpose=PURPOSE,files=before))

def verify_snapshot(path):
    m=read(path/'SNAPSHOT_COMPLETE.json');actual=inventory(path);actual.pop('SNAPSHOT_COMPLETE.json')
    require(m['purpose']==PURPOSE and actual==m['files'],'Incomplete/corrupt evidence snapshot')

def worker(args):
    import numpy as np,tensorflow as tf
    from research.code.determinism import configure_determinism
    from research.code.data import load_journal_v1,make_tf_dataset
    from research.code.recoverable_model import build_model,VARIANT
    from research.code.recoverable_checkpoint import EpochStore
    from research.code.recoverable_runner import train_epochs
    from research.code.progress import Progress
    require(args.rows in (43,171),'Bounded fixture only')
    require(args.arm in ('A','B','C1','C2','C3'),'Engineering arm required')
    require((args.arm in ('C2','C3')) == bool(args.checkpoint and args.receipt),'Explicit continuation checkpoint/receipt required')
    require(os.environ.get('PYTHONHASHSEED')=='113' and os.environ.get('TF_DETERMINISTIC_OPS')=='1','Fresh deterministic process required')
    if not args.local_cpu_test:
        from research.tools.colab_gpu_worker import gpu_environment
        environment=gpu_environment()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    else:environment=dict(python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,device='CPU_ENGINEERING_ONLY');gpu='CPU_ENGINEERING_ONLY'
    det=configure_determinism(113);require(det.deterministic_ops_enabled and not det.warnings,'Determinism warnings')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    write_new(out/'ENVIRONMENT.json',dict(environment=environment,gpu=gpu,determinism=det.to_dict()))
    split=ROOT/'research/splits/journal_v1'
    frames=load_journal_v1(split/'train.csv',split/'validation.csv',split/'test.csv',[args.dataset/f'HAM10000_images_part_{i}' for i in (1,2)])
    identity=dict(purpose=PURPOSE,variant=VARIANT,fixture_rows=args.rows,fixture_epochs=3,source={n:sha(ROOT/n) for n in ('research/code/recoverable_runner.py','research/code/recoverable_checkpoint.py','research/code/recoverable_model.py','research/code/data.py','daa_ham10000_interpretability_fixed_v2.py')},environment=environment if args.local_cpu_test else {k:environment[k] for k in ('python','versions','freeze','tensorflow_build','float_policy')},gpu=gpu,manifests={p.name:sha(p) for p in split.glob('*.csv')},inputs={r.image_id:sha(Path(r.filepath)) for r in frames['train'].iloc[:args.rows+11].itertuples()})
    write_new(out/'IDENTITY.json',identity)
    baseline,model,cfg=build_model(ROOT,7);opt=tf.keras.optimizers.Adam(.001)
    ds=make_tf_dataset(frames['train'].iloc[:args.rows],128,32,shuffle=True,seed=113)
    vd=make_tf_dataset(frames['train'].iloc[args.rows:args.rows+11],128,32,shuffle=False,seed=None)
    store=EpochStore(out/'checkpoints',args.durable/'checkpoints',identity,args.checkpoint,args.receipt);store.segment_id=args.arm
    original=baseline.train_step
    def observed(m,o,x,y,c):
        values=original(m,o,x,y,c)
        arrays={f'model_{i}':v.numpy() for i,v in enumerate(m.variables)}
        arrays.update({f'optimizer_{i}':v.numpy() for i,v in enumerate(o.variables)})
        arrays.update(input=x.numpy(),labels=y.numpy(),rng=m.sampling_rng.state.numpy(),loss=np.array([v.numpy() for v in values]))
        np.savez_compressed(out/f'step{int(o.iterations.numpy())}.npz',**arrays)
        return values
    baseline.train_step=observed
    persist=store.persist
    def saving(*values):
        if args.arm=='A':return
        record=persist(*values)
        if args.arm in ('C1','C2'):
            write_new(out/'KILL_READY.json',record)
            while True:time.sleep(.2)
    store.persist=saving
    progress=Progress();progress.bind(out);started=time.perf_counter()
    history=train_epochs(model=model,optimizer=opt,baseline=baseline,cfg=cfg,train_dataset=ds,validation_dataset=vd,store=store,epochs=3,training_rows=args.rows,run_dir=out,progress=progress)
    write_new(out/'RESULT.json',dict(purpose=PURPOSE,history=history,updates=int(opt.iterations.numpy()),seconds=time.perf_counter()-started,checkpoint_records=store.receipts,gpu_memory=tf.config.experimental.get_memory_info('GPU:0') if not args.local_cpu_test else None,scientific_evidence=False))

def compare(case,rows):
    import numpy as np
    checks=[];steps=(rows+31)//32
    for step in range(1,steps*3+1):
        with np.load(case/'A'/f'step{step}.npz') as a:
            for arm in ('B','C'+str((step-1)//steps+1)):
                with np.load(case/arm/f'step{step}.npz') as b:
                    require(set(a.files)==set(b.files),'Tensor keys differ')
                    for name in a.files:
                        checks.append(dict(step=step,arm=arm,tensor=name,equal=np.array_equal(a[name],b[name]),max_abs=float(np.max(np.abs(a[name].astype(float)-b[name].astype(float))))))
    histories=[read(case/a/'RESULT.json')['history'] for a in ('A','B','C3')]
    require(histories[0]==histories[1]==histories[2],'Epoch metrics/history differ')
    bad=[c for c in checks if not c['equal']]
    result=dict(status='EXACT_BOUNDED_ENGINEERING' if not bad else 'EQUIVALENCE_FAILED',purpose=PURPOSE,rows=rows,epochs=3,checks=len(checks),first_divergence=bad[0] if bad else None,production_approval=False)
    write_new(case/'COMPARISON.json',dict(summary=result,checks=checks));require(not bad,'First divergence: '+str(bad[:1]));return result

def drive_faults(local,durable):
    from research.code.recoverable_checkpoint import seal,verify,transfer
    src=local/'source'/'faultfixture';src.mkdir(parents=True,exist_ok=False)
    (src/'synthetic.data').write_bytes(b'ENGINEERING FAULT FIXTURE; NOT MODEL STATE')
    write_new(src/'CONTROL.json',dict(epoch=1,step=251,phase='epoch_complete',next_operation='next_epoch_or_final_export',history=[dict(epoch=1)],prefix='state-1'))
    identity=dict(fault_fixture=True);seal(src,identity);digest=sha(src/'COMPLETE.json')
    results=[]
    for fault in ('incomplete','corrupt'):
        dest=durable/fault/src.name;transfer(src,dest,identity,digest)
        if fault=='incomplete':(dest/'COMPLETE.json').rename(dest/'UNPUBLISHED_RECEIPT.json')
        else:(dest/'synthetic.data').write_bytes(b'INTENTIONAL FAULT')
        try:verify(dest,identity,digest)
        except (ValueError,FileNotFoundError) as exc:results.append(dict(fault=fault,rejected=True,error=repr(exc)))
        else:raise RuntimeError('Corrupted engineering fixture accepted')
    write_new(local/'FAULT_RESULTS.json',results)
    snapshot(local,durable/'fault_evidence')

def orchestrate(args):
    from research.code.colab_runtime import verify_images,source_guard
    package=read(args.payload/'PACKAGE.json')
    require(package['release'] and not package['source_status'] and package['execution_scope']==SCOPE,'Clean engineering package required')
    source_guard(ROOT,package)
    verify_images(args.dataset,read(args.payload/'DATASET.json'))
    if not args.local_cpu_test:
        require(sys.platform.startswith('linux') and os.path.ismount('/content/drive'),'Mounted Colab Drive required')
        expected=Path('/content/drive/MyDrive/DeepAA_recovery_engineering')
        require(args.durable.resolve().parent==expected,'Use a unique engineering namespace')
    args.output.mkdir(parents=True,exist_ok=False);args.durable.mkdir(parents=True,exist_ok=False)
    require(shutil.disk_usage(args.output).free>3*1024**3 and shutil.disk_usage(args.durable).free>3*1024**3,'Insufficient filesystem headroom; remote Drive quota also needs operator review')
    write_new(args.output/'RUN.json',dict(purpose=PURPOSE,head=package['head'],package_sha256=sha(args.payload/'PACKAGE.json'),dataset_inventory_sha256=sha(args.payload/'DATASET.json'),dataset_verification='ALL_JPEG_AND_METADATA_BYTES_MATCH',primary_training=False,resources=dict(local_free_bytes=shutil.disk_usage(args.output).free,drive_filesystem_free_bytes=shutil.disk_usage(args.durable).free,remote_quota='OPERATOR_MUST_VERIFY',memory=Path('/proc/meminfo').read_text() if sys.platform.startswith('linux') else 'WINDOWS_CPU_ONLY'),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if sys.platform.startswith('linux') else 'WINDOWS_CPU_ONLY'))
    env=dict(os.environ,PYTHONHASHSEED='113',TF_DETERMINISTIC_OPS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    if args.local_cpu_test:env['CUDA_VISIBLE_DEVICES']='-1'
    drive_faults(args.output/'faults',args.durable/'faults')
    summaries=[]
    for rows in (43,171):
        case=args.output/str(rows);case.mkdir();last=None
        for arm in ('A','B','C1','C2','C3'):
            cmd=[sys.executable,'-B',__file__,'worker','--rows',str(rows),'--arm',arm,'--dataset',str(args.dataset),'--output',str(case/arm),'--durable',str(args.durable/str(rows))]
            if args.local_cpu_test:cmd+=['--local-cpu-test']
            if arm in ('C2','C3'):cmd+=['--checkpoint',last['path'],'--receipt',last['receipt_sha256']]
            with (case/(arm+'.log')).open('x',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
                marker=case/arm/'KILL_READY.json'
                while proc.poll() is None:
                    if marker.exists():
                        last=read(marker);proc.kill();code=proc.wait();write_new(case/arm/'KILLED.json',dict(pid=proc.pid,exit_code=code,checkpoint=last));break
                    if time.monotonic()-start>1200:
                        proc.kill();proc.wait();raise RuntimeError('Bounded worker timeout; preserve logs')
                    time.sleep(.2)
                if arm not in ('C1','C2'):require(proc.returncode==0,'Worker failed; inspect '+str(case/(arm+'.log')))
                else:require((case/arm/'KILLED.json').exists(),'Worker failed before verified checkpoint')
            snapshot(case/arm,args.durable/str(rows)/'evidence'/arm)
        summaries.append(compare(case,rows))
    write_new(args.output/'ENGINEERING_RESULT.json',dict(status='LOCAL_CPU_ENGINEERING_ONLY' if args.local_cpu_test else 'GPU_ENGINEERING_RESULTS_PENDING_REVIEW',comparisons=summaries,primary_training=False,actual_runtime_recovery='NOT_YET_TESTED',hardware_policy='same GPU UUID; cross-device continuation not approved'))
    snapshot(args.output,args.durable/'complete_evidence')
    print(json.dumps(read(args.output/'ENGINEERING_RESULT.json'),indent=2))

def recover_session(args):
    from research.code.colab_runtime import verify_images,source_guard
    require(sys.platform.startswith('linux') and os.path.ismount('/content/drive'),'Fresh mounted Colab runtime required')
    package=read(args.payload/'PACKAGE.json')
    require(package['release'] and not package['source_status'] and package['execution_scope']==SCOPE,'Engineering source only')
    source_guard(ROOT,package);verify_images(args.dataset,read(args.payload/'DATASET.json'))
    original=args.durable/'complete_evidence';verify_snapshot(original)
    old=read(original/'RUN.json')
    require(old['package_sha256']==sha(args.payload/'PACKAGE.json'),'Original engineering package differs')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(old['boot_id']!=boot,'Fresh-runtime test requires a different boot ID')
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONHASHSEED='113',TF_DETERMINISTIC_OPS='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    summaries=[]
    for rows in (43,171):
        case=args.output/str(rows);case.mkdir()
        for arm in ('A','B','C1'):
            src=args.durable/str(rows)/'evidence'/arm;verify_snapshot(src)
            shutil.copytree(src,case/arm)
        last=read(case/'C1'/'KILL_READY.json')
        for arm in ('C2','C3'):
            cmd=[sys.executable,'-B',__file__,'worker','--rows',str(rows),'--arm',arm,'--dataset',str(args.dataset),'--output',str(case/arm),'--durable',str(args.durable/('fresh_'+args.output.name)/str(rows)),'--checkpoint',last['path'],'--receipt',last['receipt_sha256']]
            with (case/(arm+'.log')).open('x',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic();marker=case/arm/'KILL_READY.json'
                while proc.poll() is None:
                    if marker.exists():
                        last=read(marker);proc.kill();code=proc.wait();write_new(case/arm/'KILLED.json',dict(exit_code=code));break
                    if time.monotonic()-start>1200:proc.kill();proc.wait();raise RuntimeError('Bounded worker timeout')
                    time.sleep(.2)
                require((case/arm/'KILLED.json').exists() if arm=='C2' else proc.returncode==0,'Fresh-runtime worker rejected checkpoint/environment; preserve log')
        summaries.append(compare(case,rows))
    write_new(args.output/'FRESH_RUNTIME_RESULT.json',dict(status='FRESH_RUNTIME_ENGINEERING_PENDING_REVIEW',original_boot=old['boot_id'],current_boot=boot,comparisons=summaries,primary_approval=False))
    snapshot(args.output,args.durable/('fresh_'+args.output.name)/'evidence')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['worker','run','compare','recover-session']);p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--durable',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--rows',type=int,choices=[43,171]);p.add_argument('--arm',choices=['A','B','C1','C2','C3']);p.add_argument('--checkpoint',type=Path);p.add_argument('--receipt');p.add_argument('--local-cpu-test',action='store_true');a=p.parse_args()
    if a.mode=='worker':worker(a)
    elif a.mode=='run':orchestrate(a)
    elif a.mode=='recover-session':recover_session(a)
    else:print(compare(a.output,a.rows))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('ENGINEERING FAILED; preserve local logs and all incomplete durable objects: '+repr(exc),file=sys.stderr,flush=True)
        raise
