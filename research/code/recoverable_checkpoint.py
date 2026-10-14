"""Epoch-boundary coherent checkpoints. No implicit latest, retries or claim edits."""
import json,os,random,shutil,time,uuid
from pathlib import Path
from .workflow_evidence import require,read,sha,write_new

def files(path):
    entries={}
    for p in sorted(path.rglob('*')):
        require(not p.is_symlink(),'Checkpoint symlink')
        if p.is_file() and p.name not in ('MANIFEST.json','COMPLETE.json'):
            entries[p.relative_to(path).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return entries

def seal(path,identity):
    inv=files(path);require(bool(inv),'Empty checkpoint')
    write_new(path/'MANIFEST.json',dict(schema='epoch_checkpoint_v1',checkpoint_id=path.name,identity=identity,files=inv))
    require(files(path)==inv,'Checkpoint changed while sealing')
    write_new(path/'COMPLETE.json',dict(manifest_sha256=sha(path/'MANIFEST.json'),status='COMPLETE_CHECKPOINT_NOT_RUN_COMPLETION'))
    verify(path,identity,sha(path/'COMPLETE.json'))

def checkpoint_budget(identity):
    if identity.get('purpose') == 'BOUNDED_RECOVERY_ENGINEERING':
        rows=identity.get('fixture_rows')
        require(type(rows) is int and rows in (43,171),'Engineering fixture scope')
        require(identity.get('fixture_epochs')==3,'Engineering epoch scope')
        return 3,(rows+31)//32
    return 50,251

def verify(path,identity,receipt_sha):
    path=Path(path)
    require(sha(path/'COMPLETE.json')==receipt_sha,'Unreviewed checkpoint receipt')
    receipt=read(path/'COMPLETE.json');require(receipt['status']=='COMPLETE_CHECKPOINT_NOT_RUN_COMPLETION','Checkpoint receipt status')
    require(sha(path/'MANIFEST.json')==receipt['manifest_sha256'],'Manifest hash')
    m=read(path/'MANIFEST.json');require(m['schema']=='epoch_checkpoint_v1' and m['checkpoint_id']==path.name,'Checkpoint identity')
    require(m['identity']==identity,'Source/data/environment/attempt mismatch')
    require(files(path)==m['files'],'Checkpoint inventory mismatch')
    c=read(path/'CONTROL.json');require(c['phase']=='epoch_complete' and c['next_operation']=='next_epoch_or_final_export','Checkpoint phase')
    limit,steps=checkpoint_budget(identity)
    require(type(c['epoch']) is int and 1<=c['epoch']<=limit,'Epoch budget')
    require(c['step']==c['epoch']*steps,'Optimizer budget')
    require([r['epoch'] for r in c['history']]==list(range(1,c['epoch']+1)),'History continuity')
    require(c['prefix']=='state-1','Explicit checkpoint prefix')
    return c

def transfer(source,destination,identity,receipt_sha):
    verify(source,identity,receipt_sha);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    for p in sorted(Path(source).rglob('*')):
        if not p.is_file() or p.name=='COMPLETE.json':continue
        target=destination/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        with p.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
        require(sha(target)==sha(p),'Durable readback failed')
    # Receipt is last; partial objects are preserved but never resumable.
    with (destination/'COMPLETE.json').open('xb') as f:
        f.write((Path(source)/'COMPLETE.json').read_bytes());f.flush();os.fsync(f.fileno())
    verify(destination,identity,receipt_sha)
    return destination

def rng_state():
    import numpy as np
    n=np.random.get_state()
    return dict(python=json.loads(json.dumps(random.getstate())),numpy=[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])])
def restore_rng(value):
    import numpy as np
    def tup(v):return tuple(tup(x) for x in v) if isinstance(v,list) else v
    random.setstate(tup(value['python']));n=value['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    require(rng_state()==value,'RNG restoration mismatch')

class EpochStore:
    def __init__(self,local,durable,identity,restore=None,reviewed_sha=None):
        self.local=Path(local);self.durable=Path(durable);self.identity=identity
        self.restore=Path(restore) if restore else None;self.reviewed_sha=reviewed_sha
        self.receipts=[]
    def load(self,model,optimizer,iterator):
        if self.restore is None:return [],[]
        import tensorflow as tf
        c=verify(self.restore,self.identity,self.reviewed_sha)
        model(tf.zeros((1,128,128,3)),training=False);optimizer.build(model.trainable_variables)
        ck=tf.train.Checkpoint(model=model,optimizer=optimizer,rng=model.sampling_rng,iterator=iterator)
        ck.restore(str(self.restore/c['prefix'])).assert_consumed()
        require(int(optimizer.iterations.numpy())==c['step'],'Restored optimizer iterations')
        restore_rng(c['rng']);return c['history'],c['progress']
    def persist(self,model,optimizer,iterator,epoch,history,progress):
        import tensorflow as tf
        limit,steps=checkpoint_budget(self.identity)
        require(1<=epoch<=limit and int(optimizer.iterations.numpy())==epoch*steps,'Epoch/update mismatch')
        start=time.monotonic();path=self.local/uuid.uuid4().hex;path.mkdir(parents=True,exist_ok=False)
        ck=tf.train.Checkpoint(model=model,optimizer=optimizer,rng=model.sampling_rng,iterator=iterator)
        prefix=ck.save(str(path/'state'));require(Path(prefix).name=='state-1','Checkpoint prefix')
        write_new(path/'CONTROL.json',dict(epoch=epoch,step=int(optimizer.iterations.numpy()),phase='epoch_complete',next_operation='next_epoch_or_final_export',prefix='state-1',history=history,progress=progress,rng=rng_state()))
        write_new(path/'ORIGIN.json',dict(segment_id=self.segment_id))
        seal(path,self.identity);digest=sha(path/'COMPLETE.json')
        dest=transfer(path,self.durable/path.name,self.identity,digest)
        record=dict(epoch=epoch,path=str(dest),receipt_sha256=digest,seconds=time.monotonic()-start,files=files(dest))
        self.receipts.append(record);print('DURABLE_EPOCH '+json.dumps(record),flush=True)
        return record
