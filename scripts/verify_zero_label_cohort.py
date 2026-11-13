"""Independent paired-sample and train-only scaling audit of saved experiments.

Does not import the model or preprocessing implementation. Training/calibration
target files are never opened. Graph methods share evaluation accounts; graph
context nodes remain an explicitly different information budget.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
p.add_argument('--data',type=Path,required=True);p.add_argument('--graph-data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
records=json.loads((args.runs/'records.json').read_text());assert len(records)==144
groups=defaultdict(list)
for r in records:groups[r['track'],r['dataset'],r['seed']].append(r)
assert len(groups)==33
def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(2**20):h.update(b)
    return h.hexdigest()
checked=[]
for (track,dataset,seed),group in sorted(groups.items()):
    assert len(group)==(4 if track=='tabular' else 6)
    indices=None;pre=None
    for r in group:
        root=args.runs/track/dataset/r['method']/str(seed)
        with np.load(root/'indices.npz',allow_pickle=False) as z:current={k:z[k] for k in z.files}
        with np.load(root/'preprocessor.npz',allow_pickle=False) as z:scaler={k:z[k] for k in z.files}
        if indices is None:indices,pre=current,scaler
        else:
            assert current.keys()==indices.keys()
            for k in indices:np.testing.assert_array_equal(indices[k],current[k])
            assert scaler.keys()==pre.keys()
            for k in pre:np.testing.assert_array_equal(pre[k],scaler[k])
        if r['method'] in ('cae_if_core','ae'):assert r['model_info']['steps']==400
        if r['method']=='audit_hcl_zero_adapted':assert r['model_info']['steps']==200
        if r['method']!='ebgad_release_zero':assert r['checks']['checkpoint_max_abs']<=1e-4
    if track=='tabular':
        data=args.data/dataset;meta=json.loads((data/'manifest.json').read_text())
        matrix=np.memmap(data/'train.x.f32',dtype='float32',mode='r',shape=(meta['splits']['train']['rows'],len(meta['features'])))
        ix=indices['train'];assert len(ix)==min(8192,len(matrix)) and len(ix)==len(np.unique(ix))
        assert (ix>=0).all() and (ix<len(matrix)).all();x=np.asarray(matrix[ix],dtype='float64')
        ci=indices['calibration'];assert len(ci)==min(100000,meta['splits']['calibration']['rows'])
        assert len(ci)==len(np.unique(ci)) and (ci>=0).all() and (ci<meta['splits']['calibration']['rows']).all()
        target=data/'test.y.i8'
    else:
        data=args.graph_data/dataset
        with np.load(data/'train.graph.npz',allow_pickle=False) as z:
            raw=z['x'];x=raw.reshape(-1,raw.shape[-1]).astype('float64')
            np.testing.assert_array_equal(indices['train'],np.flatnonzero(raw[0,:,0]>.5))
        for split in ('calibration','test'):
            with np.load(data/f'{split}.graph.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(indices[split],np.flatnonzero(z['x'][0,:,0]>.5))
        target=data/'test.y.npy'
    assert np.isfinite(x).all()
    assert pre['keep'].all() and not pre['missing'].any()
    center=np.median(x,axis=0);q25,q75=np.quantile(x,[.25,.75],axis=0);std=x.std(axis=0)
    iqr=(q75-q25)/1.349;scale=np.where(iqr>1e-8,iqr,np.where(std>1e-8,std,1.))
    np.testing.assert_allclose(pre['center'],center,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(pre['scale'],scale,rtol=1e-12,atol=1e-12)
    target_sha=file_hash(target)
    assert all(r['checks']['test_target_sha256']==target_sha for r in group)
    checked.append(dict(track=track,dataset=dataset,seed=seed,methods=len(group),
        shared_indices={k:hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k,v in indices.items()},
        shared_preprocessing_recomputed_from_training_only=True,test_target_sha256=target_sha))
args.output.write_text(json.dumps(dict(status='passed',groups=len(checked),runs=len(records),
    notes='The audit verifies the shared train-only robust scaler and paired graph evaluation accounts. EB-GAD additionally re-standardizes each evaluation graph and fits its EB parameters transductively; algorithm-internal transforms are not claimed train-only.',
    verified=checked),indent=2)+'\n')
print('ZERO_LABEL_COHORT_AND_TRAIN_ONLY_SCALING_VERIFIED',flush=True)
