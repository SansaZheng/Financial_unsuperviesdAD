"""Verify immutable local output evidence independently of metric helpers."""
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .classical import ClassicalDetector, METHODS as CLASSICAL
from .data import Cache, sha256
from .neural import NeuralDetector
from .preprocessing import Preprocessor
from .report import collect
from .detectors import load_detector, extension


def verify(runs,cache_root):
    env,records=collect(runs)
    torch.set_num_threads(2)
    provenance=Path(runs)/'cache_files_sha256.json'
    input_files=0
    if provenance.exists():
        for name,entry in json.loads(provenance.read_text()).items():
            path=Path(cache_root)/name
            assert path.stat().st_size==entry['bytes'] and sha256(path)==entry['sha256']
            input_files+=1
    total=0
    adaptation_checks=0;unchanged_base_checks=0;rgld_strict_checks=0
    shared_indices={}
    for result in records:
        root=Path(runs)/result['dataset']/f'{result["method"]}_seed{result["seed"]}'
        cache=Cache(cache_root,result['dataset'])
        score_file=result.get('test_scores_file','test_scores.npy')
        assert sha256(root/score_file)==result['test_scores_sha256']
        assert sha256(cache.path/'manifest.json')==result['run_config']['data_manifest_sha256']
        if score_file.endswith('.npz'):
            with np.load(root/score_file,allow_pickle=False) as archive:scores=archive['scores']
        else:scores=np.load(root/score_file,mmap_mode='r')
        y=cache.y('test')
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
            key=(result['dataset'],result['seed'],split)
            if key in shared_indices:
                np.testing.assert_array_equal(indices,shared_indices[key])
            else:
                shared_indices[key]=indices
        # Recompute spread-out observations from the actual saved checkpoints.
        # CPU/MPS kernels may differ by float32 rounding, so disclose tolerance.
        indices=np.linspace(0,len(scores)-1,min(32,len(scores)),dtype=np.int64)
        prep=Preprocessor.load(root/'preprocessor.npz')
        model=load_detector(result['method'],root/f'model.{extension(result["method"])}',device='cpu')
        if result['method']=='rttad2026':
            adaptation_indices=np.load(root/'adaptation_indices.npy')
            expected=np.sort(np.random.default_rng(result['seed']+2000).choice(len(scores),min(8192,len(scores)),replace=False))
            np.testing.assert_array_equal(adaptation_indices,expected)
            assert len(adaptation_indices)==result['training']['adaptation_pool_rows']
            assert (root/'model_base.pt').exists();adaptation_checks+=1
            if not any(r.get('status')=='updated' for r in result['training']['adaptation']):
                base=load_detector(result['method'],root/'model_base.pt',device='cpu')
                assert all(torch.equal(v,base.model.state_dict()[k]) for k,v in model.model.state_dict().items())
                unchanged_base_checks+=1;del base
        with threadpool_limits(limits=2):
            reproduced=model.score(prep.transform(np.asarray(cache.x('test')[indices])))
        if result['method']=='rgld2026':
            np.testing.assert_allclose(reproduced,scores[indices],rtol=0,atol=1e-12)
            rgld_strict_checks+=1
        else:
            np.testing.assert_allclose(reproduced,scores[indices],rtol=5e-4,atol=1e-5)
        del model
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
                same_seed_sampling_shared_between_methods=True,input_files_sha256_verified=input_files,
                checkpoint_predictions_reproduced_on_cpu=dict(rows_per_run_up_to=32,rtol=5e-4,atol=1e-5),
                rgld_stricter_batch_verification=dict(verified_runs=rgld_strict_checks,rtol=0,atol=1e-12),
                adaptation_pool_indices_verified=adaptation_checks,unchanged_base_verified_when_no_update=unchanged_base_checks,
                raw_rows_in_report=False)
