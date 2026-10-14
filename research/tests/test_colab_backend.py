"""Real import checks and worker environment scoping; no training or GPU claims."""
import os,sys,json,subprocess
from pathlib import Path
import pytest
from research.code import colab_runtime as runtime

INLINE='module://matplotlib_inline.backend_inline'

@pytest.mark.parametrize('mode',['environment','benchmark','train','validate','pair_verify','aggregate'])
def test_worker_backend_scoped_without_parent_mutation(tmp_path,monkeypatch,mode):
    monkeypatch.setenv('MPLBACKEND',INLINE)
    captured={}
    class Child:
        pid=123
        def poll(self):return 0
        def wait(self):return 0
    def popen(command,**kwargs):
        captured.update(kwargs);return Child()
    monkeypatch.setattr(runtime.subprocess,'Popen',popen)
    monkeypatch.setattr(runtime,'process_snapshot',lambda:[])
    runtime.worker(tmp_path,tmp_path,mode,[],tmp_path/'worker.log',seed=113)
    assert captured['env']['MPLBACKEND']=='Agg'
    assert captured['env']['PYTHONHASHSEED']=='113'
    assert captured['env']['TF_DETERMINISTIC_OPS']=='1'
    assert os.environ['MPLBACKEND']==INLINE
    assert ('CUDA_VISIBLE_DEVICES' in captured['env']) if mode in ('validate','pair_verify','aggregate') else True

@pytest.mark.parametrize('backend',[INLINE,'Agg'])
def test_real_tensorflow_keras_import(tmp_path,backend):
    env=dict(os.environ,MPLBACKEND=backend,PYTHONDONTWRITEBYTECODE='1')
    code="import tensorflow as tf; import keras; import matplotlib; import json; print(json.dumps({'tensorflow':tf.__version__,'keras':keras.__version__,'backend':matplotlib.get_backend()}))"
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True)
    (tmp_path/'import.log').write_text(result.stdout+result.stderr)
    if backend==INLINE:
        assert result.returncode!=0
        assert "Key backend" in result.stderr and INLINE in result.stderr
    else:
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout.strip().splitlines()[-1])['backend'].lower()=='agg'
