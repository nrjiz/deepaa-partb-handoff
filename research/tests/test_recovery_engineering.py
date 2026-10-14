"""Bounded engineering isolation, persistence and shared-loop guards."""
import json
from pathlib import Path
import pytest
from research.code.recoverable_checkpoint import checkpoint_budget
from research.tools.recovery_engineering import snapshot,verify_snapshot,SCOPE
from research.code.recoverable_workflow import approval_gate
from research.tests.test_recoverable_k7 import auth
ROOT=Path(__file__).resolve().parents[2]

def test_production_budget_is_unchanged():assert checkpoint_budget({})==(50,251)
@pytest.mark.parametrize('rows,steps',[(43,2),(171,6)])
def test_engineering_budget(rows,steps):assert checkpoint_budget(dict(purpose='BOUNDED_RECOVERY_ENGINEERING',fixture_rows=rows,fixture_epochs=3))==(3,steps)
@pytest.mark.parametrize('rows,epochs',[(8011,3),(43,50),(True,3),(171,4)])
def test_refuse_expanded_fixture(rows,epochs):
 with pytest.raises(ValueError):checkpoint_budget(dict(purpose='BOUNDED_RECOVERY_ENGINEERING',fixture_rows=rows,fixture_epochs=epochs))

def test_engineering_package_refused_for_primary():
 p=dict(release=True,source_status='',head='head',execution_scope=SCOPE)
 with pytest.raises(ValueError,match='Scope mismatch'):approval_gate(p,auth(),'b'*64,'head')

def test_snapshot_readback_and_no_overwrite(tmp_path):
 a=tmp_path/'source';a.mkdir();(a/'data').write_bytes(b'abc');b=tmp_path/'durable';snapshot(a,b);verify_snapshot(b)
 with pytest.raises(ValueError):snapshot(a,b)
@pytest.mark.parametrize('fault',['missing','corrupt','receipt'])
def test_snapshot_failure_refused(tmp_path,fault):
 a=tmp_path/'source';a.mkdir();(a/'data').write_bytes(b'abc');b=tmp_path/'durable';snapshot(a,b)
 if fault=='missing':(b/'data').unlink()
 if fault=='corrupt':(b/'data').write_bytes(b'x')
 if fault=='receipt':(b/'SNAPSHOT_COMPLETE.json').unlink()
 with pytest.raises((ValueError,FileNotFoundError)):verify_snapshot(b)

def test_failed_copy_never_publishes_receipt(tmp_path,monkeypatch):
 import research.tools.recovery_engineering as m
 a=tmp_path/'source';a.mkdir();(a/'data').write_bytes(b'abc');b=tmp_path/'durable'
 def fail(*a,**k):raise OSError('injected quota/transfer failure')
 monkeypatch.setattr(m.shutil,'copyfileobj',fail)
 with pytest.raises(OSError):snapshot(a,b)
 assert b.exists() and not (b/'SNAPSHOT_COMPLETE.json').exists()

def test_epoch50_calls_no_training(tmp_path):
 from research.code.recoverable_runner import train_epochs
 class Store:
  def load(self,*a):return [{'epoch':n} for n in range(1,51)],[]
  def persist(self,*a):raise AssertionError('Must not checkpoint or update')
 class Never:
  def __getattr__(self,n):raise AssertionError(n)
 history=train_epochs(model=None,optimizer=None,baseline=Never(),cfg=None,train_dataset=[],validation_dataset=None,store=Store(),epochs=50,training_rows=8011,run_dir=tmp_path,progress=Never())
 assert len(history)==50

def test_engineering_notebook_has_no_primary_cell():
 nb=json.loads((ROOT/'research/notebooks/recovery_gpu_acceptance.ipynb').read_text())
 code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
 for c in nb['cells']:
  if c['cell_type']=='code':compile(''.join(c['source']),'notebook','exec')
 assert 'PRIMARY_LAUNCH_ENABLED = False' in code
 assert 'colab_recoverable_k7.py' not in code and 'colab_k7_single.py' not in code
 assert '--force-reinstall' in code and 'bounded recovery engineering' in code

def test_engineering_scope_refused_by_original_single_runner():
 from types import SimpleNamespace
 from research.code.colab_k7_single import K7Single
 with pytest.raises(ValueError,match='does not authorize'):K7Single.identity(SimpleNamespace(package={'execution_scope':SCOPE}))

def test_durable_fault_injection_records_refusals(tmp_path):
 from research.tools.recovery_engineering import drive_faults
 drive_faults(tmp_path/'local',tmp_path/'durable')
 records=json.loads((tmp_path/'local/FAULT_RESULTS.json').read_text())
 assert [r['fault'] for r in records]==['incomplete','corrupt'] and all(r['rejected'] for r in records)
 verify_snapshot(tmp_path/'durable/fault_evidence')
