"""JAFFE-specific filename, coverage and TIFF adapter; no lesion aliases."""
from collections import Counter
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .common import ROOT, read, require, sha, write

PARTICIPANTS = {'KA', 'KL', 'KM', 'KR', 'MK', 'NA', 'NM', 'TM', 'UY', 'YM'}
PATTERN = re.compile(r'^(KA|KL|KM|KR|MK|NA|NM|TM|UY|YM)\.(AN|DI|FE|HA|NE|SA|SU)([1-5])\.(\d+)\.tiff$')


def parse_name(name):
    m = PATTERN.fullmatch(name)
    require(m is not None, 'Unrecognized JAFFE filename: '+name)
    return dict(image_id=name, participant_id=m[1], expression=m[2], pose_number=int(m[3]), archive_number=int(m[4]))


def coverage(rows, c):
    split = c['participants']
    flat = [v for ids in split.values() for v in ids]
    require(len(flat) == len(set(flat)) == 10 and set(flat) == PARTICIPANTS, 'Participant overlap or omission')
    report = {}
    for part in ('train', 'validation', 'test'):
        selected = [r for r in rows if r['participant_id'] in split[part]]
        require(set(r['expression'] for r in selected) == set(c['classes']), 'Missing expression in '+part)
        report[part] = dict(images=len(selected), participants=split[part], expression_counts=dict(Counter(r['expression'] for r in selected)), participant_expression_counts={p:dict(Counter(r['expression'] for r in selected if r['participant_id']==p)) for p in split[part]})
    return report


def audit(data_root, c, config_sha, output):
    if c['dataset']=='HAM10000':
        return audit_ham(data_root,c,config_sha,output)
    data_root = Path(data_root).resolve()
    expected = set(read(ROOT/'jaffe_partb/official_filename_inventory.json')['filenames'])
    files = list(data_root.rglob('*.tiff'))
    require(len(files)==213 and {p.name for p in files}==expected, 'Official 213-image filename membership differs')
    rows = []
    for p in sorted(files, key=lambda p:p.name):
        require(not p.is_symlink(), 'Data symlink refused')
        with Image.open(p) as im:
            require(im.mode=='L' and im.size==(256,256) and getattr(im,'n_frames',1)==1, 'Expected single-frame 256x256 8-bit grayscale TIFF')
            im.load()
        row = parse_name(p.name)
        row.update(class_label=row['expression'],relative_path=p.relative_to(data_root).as_posix(), sha256=sha(p), bytes=p.stat().st_size,
                   partition=next(part for part,ids in c['participants'].items() if row['participant_id'] in ids))
        rows.append(row)
    result = dict(protocol=c['protocol'], experiment_id=c['experiment_id'],dataset=c['dataset'],config_sha256=config_sha, files=rows, coverage=coverage(rows,c),
                  image_bytes_verified_locally=True, official_image_hashes_available=False)
    write(output, result)
    return result


def load(manifest, data_root, c, config_sha):
    m = read(manifest)
    require(m['protocol']==c['protocol'] and m['config_sha256']==config_sha and m['experiment_id']==c['experiment_id'] and m['dataset']==c['dataset'], 'Manifest/config mismatch')
    if c['dataset']=='HAM10000':
        return load_ham(m,data_root,c)
    rows=m['files']; expected=set(read(ROOT/'jaffe_partb/official_filename_inventory.json')['filenames'])
    require(len(rows)==213 and {r['image_id'] for r in rows}==expected, 'Manifest membership')
    require(m['coverage']==coverage(rows,c), 'Coverage differs')
    root=Path(data_root).resolve()
    for row in rows:
        p=(root/row['relative_path']).resolve()
        require(p.is_relative_to(root) and not p.is_symlink(), 'Unsafe data path')
        require(parse_name(p.name)=={k:row[k] for k in parse_name(p.name)}, 'Filename identity differs')
        require(row['class_label']==row['expression'],'Expression label differs')
        require(row['participant_id'] in c['participants'][row['partition']], 'Split differs')
        require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], 'Dataset changed: '+p.name)
    return {part:[r for r in rows if r['partition']==part] for part in c['participants']}


def dataset(rows, root, seed=None, c=None):
    if c and c['dataset']=='HAM10000':
        import pandas as pd
        from research.code.data import make_tf_dataset
        frame=pd.DataFrame(rows);frame['filepath']=[str(Path(root)/r['relative_path']) for r in rows]
        frame['label']=[c['classes'].index(r['class_label']) for r in rows]
        return make_tf_dataset(frame,128,32,shuffle=seed is not None,seed=seed)
    import tensorflow as tf
    paths=[str(Path(root)/r['relative_path']) for r in rows]
    classes=['AN','DI','FE','HA','NE','SA','SU']
    ds=tf.data.Dataset.from_tensor_slices((paths,[classes.index(r['expression']) for r in rows]))
    if seed is not None:
        ds=ds.shuffle(min(len(rows),5000),seed=seed,reshuffle_each_iteration=True)
    def decode(path):
        with Image.open(path.numpy().decode()) as im:
            require(im.mode=='L' and im.size==(256,256), 'TIFF format changed')
            return np.repeat(np.asarray(im,dtype=np.uint8)[...,None],3,axis=2)
    def preprocess(path,label):
        x=tf.py_function(decode,[path],Tout=tf.uint8);x.set_shape((256,256,3))
        return tf.cast(tf.image.resize(x,(128,128),method='bilinear'),tf.float32)/255.,label
    ds=ds.map(preprocess,num_parallel_calls=1,deterministic=True).batch(32,drop_remainder=False).prefetch(1)
    options=tf.data.Options();options.experimental_deterministic=True
    return ds.with_options(options)


def ham_frames(root):
    from research.code.data import load_journal_v1
    split=ROOT/'research/splits/journal_v1';root=Path(root)
    return load_journal_v1(split/'train.csv',split/'validation.csv',split/'test.csv',
        [root/'HAM10000_images_part_1',root/'HAM10000_images_part_2'])


def audit_ham(root,c,config_sha,output):
    """Reuse frozen HAM membership/preprocessing; create only a new experiment manifest."""
    from research.code.constants import MANIFEST_SHA256
    root=Path(root).resolve();frames=ham_frames(root);rows=[]
    for part,frame in frames.items():
        for item in frame.to_dict('records'):
            p=Path(item['filepath']).resolve()
            require(p.is_relative_to(root) and not p.is_symlink(),'HAM data path')
            rows.append(dict(image_id=item['image_id'],lesion_id=item['lesion_id'],dx=item['dx'],class_label=item['dx'],
                relative_path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p),partition=part))
    metadata=root/'HAM10000_metadata.csv';require(metadata.is_file(),'HAM metadata required')
    result=dict(protocol=c['protocol'],experiment_id=c['experiment_id'],dataset='HAM10000',config_sha256=config_sha,
        frozen_manifest_hashes=MANIFEST_SHA256,files=rows,metadata=dict(relative_path=metadata.name,sha256=sha(metadata)),
        coverage={p:dict(images=len(f),lesions=int(f['lesion_id'].nunique()),diagnosis_counts=f['dx'].value_counts().to_dict()) for p,f in frames.items()},
        image_bytes_verified_locally=True,official_image_hashes_available=False)
    write(output,result);return result


def load_ham(m,root,c):
    from research.code.constants import MANIFEST_SHA256
    root=Path(root).resolve();frames=ham_frames(root)
    require(m['frozen_manifest_hashes']==MANIFEST_SHA256,'Frozen HAM manifest identity')
    require(sha(root/m['metadata']['relative_path'])==m['metadata']['sha256'],'Metadata changed')
    result={}
    for part,frame in frames.items():
        rows=[r for r in m['files'] if r['partition']==part]
        require([(r['image_id'],r['lesion_id'],r['dx']) for r in rows]==list(frame[['image_id','lesion_id','dx']].itertuples(index=False,name=None)),'HAM manifest order/membership differs')
        for row in rows:
            p=(root/row['relative_path']).resolve()
            require(p.is_relative_to(root) and not p.is_symlink() and row['class_label']==row['dx'],'HAM path/label')
            require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'HAM bytes changed')
        result[part]=rows
    require(sum(map(len,result.values()))==len(m['files'])==10015,'HAM full membership')
    return result
