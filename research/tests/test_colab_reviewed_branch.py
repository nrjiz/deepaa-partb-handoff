"""Real Git branch checks; no training or release approval."""
from pathlib import Path
import subprocess
import pytest
from research.code.colab_transfer import checkout_reviewed_branch,git

@pytest.fixture
def repo(tmp_path):
    p=tmp_path/'repo';p.mkdir()
    subprocess.run(['git','init',str(p)],check=True)
    subprocess.run(['git','config','user.name','Fixture'],cwd=p,check=True)
    subprocess.run(['git','config','user.email','fixture@example.invalid'],cwd=p,check=True)
    (p/'file').write_text('first\n')
    subprocess.run(['git','add','file'],cwd=p,check=True)
    subprocess.run(['git','commit','-m','fixture'],cwd=p,check=True)
    return p,git(p,'rev-parse','HEAD')

def test_absent_branch(repo):
    p,head=repo;checkout_reviewed_branch(p,head)
    assert git(p,'branch','--show-current')=='colab-reviewed'
    assert git(p,'rev-parse','HEAD')==head
    assert not git(p,'status','--porcelain')

def test_existing_correct_branch(repo):
    p,head=repo
    subprocess.run(['git','branch','colab-reviewed',head],cwd=p,check=True)
    checkout_reviewed_branch(p,head)
    assert git(p,'rev-parse','HEAD')==head
    assert not git(p,'status','--porcelain')

def test_existing_wrong_branch_unchanged(repo):
    p,first=repo
    subprocess.run(['git','branch','colab-reviewed',first],cwd=p,check=True)
    (p/'file').write_text('second\n')
    subprocess.run(['git','commit','-am','second fixture'],cwd=p,check=True)
    second=git(p,'rev-parse','HEAD')
    with pytest.raises(ValueError,match='differs'):checkout_reviewed_branch(p,second)
    assert git(p,'rev-parse','refs/heads/colab-reviewed')==first
    assert git(p,'rev-parse','HEAD')==second
