"""Identity, immutable output, inventory and approval helpers."""
import contextlib
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def plain(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)
        f.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def inventory(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        require(not p.is_symlink(), 'Symlink refused')
        if p.is_file() and p.name != 'INVENTORY.json':
            result[p.relative_to(root).as_posix()] = dict(bytes=p.stat().st_size, sha256=sha(p))
    return result


def seal(root):
    write(Path(root)/'INVENTORY.json', inventory(root))


def verify(root):
    require(read(Path(root)/'INVENTORY.json') == inventory(root), 'Artifact inventory mismatch')


def external(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(ROOT), 'Keep JAFFE data/results outside the Git checkout')
    return path


def config(path):
    c = read(path)
    require(c['protocol'] == 'partb_new_experiment_v1', 'Separate experiment schema required')
    require(re.fullmatch(r'[a-z][a-z0-9_]{3,79}',c['experiment_id']) and c['experiment_id'] not in ('archetype_stability_v1','stability_v1','journal_v1'), 'New experiment identifier required')
    require(c['dataset'] in ('JAFFE','HAM10000'), 'Dataset identity')
    require(isinstance(c['K'],list) and len(c['K'])>0 and len(set(c['K']))==len(c['K']) and all(type(k) is int and 2<=k<=10 for k in c['K']), 'Explicit distinct K values from 2 through 10 required')
    require(isinstance(c['seeds'],list) and len(c['seeds'])>=2 and len(set(c['seeds']))==len(c['seeds']) and all(type(s) is int and 0<=s<2**31 for s in c['seeds']), 'Explicit distinct seeds required')
    require(c['reference_seed'] in c['seeds'] and c['epochs'] == 50 and c['batch_size'] == 32, 'Reference/budget proposal')
    require(c['learning_rate'] == .001 and c['sampling'] == 'original_stochastic', 'Model proposal differs')
    if c['dataset']=='JAFFE':
        require(c['preprocessing'] == 'TIFF_L_repeat_RGB_tf_bilinear128_div255_no_augmentation', 'Preprocessing proposal differs')
        require(c['evaluation'] == 'image_metrics_participant_cluster_bootstrap', 'Evaluation proposal differs')
        require(c['classes'] == ['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'], 'Expression order')
    else:
        require(c['preprocessing']=='JPEG_RGB_tf_bilinear128_div255_no_augmentation' and c['evaluation']=='frozen_HAM_lesion_metrics_bootstrap' and c['split']=='journal_v1', 'New HAM experiment must retain explicit frozen split/evaluation')
        require(c['classes']==['akiec','bcc','bkl','df','mel','nv','vasc'], 'Diagnosis order')
    require(c['loss_weights'] == dict(reconstruction=1.,classification=1.,kl=.01,archetype_simplex=1.,coefficient_entropy=0.), 'Loss weights differ')
    require(c['latent_dimension']=='K-1' and c['matching']=='validation_only_Pearson_or_2MAE', 'Architecture/matching proposal differs')
    require((c['degenerate_std_threshold'],c['bootstrap_resamples'],c['bootstrap_seed'],c['nearest_real_top_n'])==(1e-8,1000,2026,10), 'Evaluation constants differ')
    return c


def approval(path, config_path, manifest_path, k, seed):
    a = read(path)
    require(a.get('status') == 'ACCEPTED' and a.get('reviewer') and a.get('accepted_utc'), 'Scientific approval pending')
    require(a.get('config_sha256') == sha(config_path), 'Approval/config mismatch')
    require(a.get('manifest_sha256') == sha(manifest_path), 'Approval/manifest mismatch')
    require(a.get('source_head') == git('rev-parse', 'HEAD'), 'Approval/source mismatch')
    require(a.get('dataset_terms_accepted') is True, 'Dataset terms must be accepted')
    require(a.get('evaluation_limitations_accepted') is True, 'Uncertainty limitations must be accepted')
    require([k, seed] in a.get('authorized_attempts', []), 'Attempt not authorized')
    require(not git('status', '--porcelain', '--untracked-files=all'), 'Clean checkout required')
    return a


@contextlib.contextmanager
def lock(root):
    """OS-released process lock; permanent attempt claims are separate and never removed."""
    p = Path(root)/'.execution.lock'
    with p.open('a+b') as f:
        f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
