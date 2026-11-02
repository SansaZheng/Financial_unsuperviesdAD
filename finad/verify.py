"""Verify immutable local output evidence independently of metric helpers."""
import json
from pathlib import Path
import numpy as np

from .data import Cache, sha256
from .report import collect


def verify(runs,cache_root):
    env,records=collect(runs)
    total=0
    for result in records:
        root=Path(runs)/result['dataset']/f'{result["method"]}_seed{result["seed"]}'
        cache=Cache(cache_root,result['dataset'])
        assert sha256(root/'test_scores.npy')==result['test_scores_sha256']
        assert sha256(cache.path/'manifest.json')==result['run_config']['data_manifest_sha256']
        scores=np.load(root/'test_scores.npy',mmap_mode='r');y=cache.y('test')
        assert scores.shape==y.shape and np.isfinite(scores).all()
        thresholds=json.loads((root/'thresholds.json').read_text())
        assert thresholds==result['thresholds']
        cal=np.load(root/'calibration_scores.npy')
        if thresholds['pot']['status']=='ok':
            pot=thresholds['pot']
            assert len(cal)==pot['n_calibration']
            assert np.count_nonzero(cal.astype(np.float64)>pot['initial_threshold'])==pot['n_peaks']
        for split,file in [('train','train_indices.npy'),('calibration','calibration_indices.npy')]:
            indices=np.load(root/file)
            assert len(indices)==len(np.unique(indices))
            assert (indices>=0).all() and (indices<cache.manifest['splits'][split]['rows']).all()
        for policy,value in thresholds.items():
            op=result['metrics']['operating_points'][policy]
            if value['status']!='ok':
                assert op is None
                continue
            if result['metrics']['evaluated']==0:
                assert op is None
                continue
            counts=dict(tp=0,fp=0,fn=0,tn=0)
            for i in range(0,len(y),500000):
                label=y[i:i+500000];alarm=scores[i:i+500000]>=value['threshold']
                counts['tp']+=int(np.count_nonzero(alarm&(label==1)))
                counts['fp']+=int(np.count_nonzero(alarm&(label==0)))
                counts['fn']+=int(np.count_nonzero(~alarm&(label==1)))
                counts['tn']+=int(np.count_nonzero(~alarm&(label==0)))
            assert all(counts[k]==op[k] for k in counts)
            denom=2*counts['tp']+counts['fp']+counts['fn']
            assert np.isclose(op['f1'],2*counts['tp']/denom if denom else 0.)
        total+=1
    return dict(verified_runs=total,score_sha256_verified=True,confusion_matrices_independently_recomputed=True,
                sample_indices_unique_and_in_bounds=True,pot_peak_counts_match_scores=True,
                raw_rows_in_report=False)
