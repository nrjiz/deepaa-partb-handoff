"""Offline real pip downgrade regression in fresh disposable Python environments."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import pytest

def approved_wheels():
    supplied=os.environ.get('ST04H_WHEELS')
    if supplied is None:
        pytest.skip('Real-pip reproduction requires ST04H_WHEELS with approved offline wheels; not verified by this run')
    wheels=Path(supplied)
    expected={'pip-25.2-py3-none-any.whl':'6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717',
              'pip-26.2.1-py3-none-any.whl':'71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e'}
    assert supplied and wheels.is_dir(), 'Provided ST04H_WHEELS must be a valid directory'
    for name,digest in expected.items():
        path=wheels/name
        assert path.is_file(), 'Missing approved offline wheel: '+name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Unapproved offline wheel hash: '+name
    return wheels.resolve()

@pytest.mark.parametrize('flag',['--ignore-installed','--force-reinstall'])
def test_real_pip_install_check_and_integrity(tmp_path,flag):
    wheels=approved_wheels()
    env=tmp_path/'environment'
    subprocess.run([sys.executable,'-B','-m','venv',str(env)],check=True)
    python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(wheels),'pip==26.2.1'],check=True)
    report=tmp_path/'installation.json'
    subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(wheels),flag,'--report',str(report),'pip==25.2'],check=True)
    check=subprocess.run([str(python),'-m','pip','check'],capture_output=True,text=True)
    if flag=='--ignore-installed':
        assert check.returncode!=0
        assert 'BuildDependencyInstallError' in check.stderr
    else:
        assert check.returncode==0,check.stderr
        info=subprocess.check_output([str(python),'-c',"import pip,importlib.metadata as m,json; print(json.dumps([pip.__version__,[d.version for d in m.distributions() if d.metadata['Name'].lower()=='pip']]))"],text=True)
        assert json.loads(info)==['25.2',['25.2']]
        site=env/('Lib/site-packages' if os.name=='nt' else 'lib/python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'/site-packages')
        assert not (site/'pip/_internal/build_env.py').exists() or 'BuildDependencyInstallError' not in (site/'pip/_internal/build_env.py').read_text()
        item=json.loads(report.read_text())['install'][0]
        assert item['metadata']['version']=='25.2'
        assert item['download_info']['archive_info']['hashes']['sha256']==hashlib.sha256((wheels/'pip-25.2-py3-none-any.whl').read_bytes()).hexdigest()

def test_notebook_pins_report_hashes_and_launch_unchanged():
    root=Path(__file__).resolve().parents[2]
    nb=json.loads((root/'research/notebooks/stability_v1_colab.ipynb').read_text())
    cell=''.join(nb['cells'][5]['source'])
    assert '--ignore-installed' not in cell and '--force-reinstall' in cell
    assert "'-m', 'pip', 'check'" in cell and "['hashes']['sha256']" in cell
    assert 'pip==25.2' in (root/'research/requirements_colab.txt').read_text()
    assert 'PRIMARY_LAUNCH_ENABLED = False' in ''.join(nb['cells'][13]['source'])

def test_missing_wheel_variable_explicit_skip(monkeypatch):
    monkeypatch.delenv('ST04H_WHEELS',raising=False)
    with pytest.raises(pytest.skip.Exception,match='not verified'):approved_wheels()

@pytest.mark.parametrize('mode',['empty','absent_directory','missing_files','bad_hash'])
def test_invalid_provided_wheels_fail(tmp_path,monkeypatch,mode):
    folder=tmp_path/'wheels'
    if mode in ('missing_files','bad_hash'):folder.mkdir()
    if mode=='bad_hash':
        (folder/'pip-25.2-py3-none-any.whl').write_bytes(b'not approved')
        (folder/'pip-26.2.1-py3-none-any.whl').write_bytes(b'not approved')
    monkeypatch.setenv('ST04H_WHEELS','' if mode=='empty' else str(folder))
    with pytest.raises(AssertionError):approved_wheels()
