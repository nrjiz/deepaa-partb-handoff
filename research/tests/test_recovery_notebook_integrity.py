"""Notebook source attestation contracts; complements actual nbclient rehearsal."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/recovery_gpu_acceptance.ipynb'


def load_contract():
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    tree = ast.parse(''.join(nb['cells'][3]['source']))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cell_sources')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return nb, namespace['cell_sources']


def test_autosave_changes_hash_but_preserves_attested_sources():
    reviewed, sources = load_contract()
    live = copy.deepcopy(reviewed)
    live['metadata']['colab']['last_runtime'] = 'simulated-runtime'
    for index, cell in enumerate(live['cells']):
        cell['metadata'].update(id=str(index), execution={'start_time': '2026-10-14'})
        cell['source'] = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            cell['execution_count'] = index
            cell['outputs'] = [dict(output_type='stream', name='stdout', text='simulated output\n')]
    assert hashlib.sha256(json.dumps(reviewed).encode()).digest() != hashlib.sha256(json.dumps(live).encode()).digest()
    assert sources(live) == sources(reviewed)


@pytest.mark.parametrize('change', ['code', 'markdown', 'order', 'insert', 'delete', 'type'])
def test_changed_cell_sources_rejected(change):
    reviewed, sources = load_contract()
    live = copy.deepcopy(reviewed)
    if change == 'code': live['cells'][1]['source'].append('PRIMARY_LAUNCH_ENABLED = True\n')
    if change == 'markdown': live['cells'][0]['source'].append('altered\n')
    if change == 'order': live['cells'][1], live['cells'][3] = live['cells'][3], live['cells'][1]
    if change == 'insert': live['cells'].append(copy.deepcopy(live['cells'][0]))
    if change == 'delete': live['cells'].pop()
    if change == 'type': live['cells'][1]['cell_type'] = 'markdown'
    assert sources(live) != sources(reviewed)


def test_attachments_rejected():
    nb, sources = load_contract()
    nb['cells'][0]['attachments'] = {'extra': {'text/plain': 'unreviewed'}}
    with pytest.raises(AssertionError): sources(nb)


@pytest.mark.parametrize('value', ['', '../escape', 'with space', 'a'*65, 'é', 'Valid113'])
def test_runtime_id_validation(value):
    nb, _ = load_contract()
    tree = ast.parse(''.join(nb['cells'][1]['source']))
    nodes = [n for n in tree.body if
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ENGINEERING_ID' for t in n.targets)) or
             (isinstance(n, ast.Assert) and 'ENGINEERING_ID' in ast.unparse(n.test))]
    namespace = {'input': lambda prompt: value}
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec')
    if value == 'Valid113':
        exec(code, namespace)
        assert namespace['ENGINEERING_ID'] == value
    else:
        with pytest.raises(AssertionError): exec(code, namespace)
