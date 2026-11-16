"""Score all fixed variants first; evaluate labels only in a separate stage."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from finad.preprocessing import Preprocessor
from finad.evaluation import metrics
from .stable import VARIANTS, METHODS, fit_prior, errors, fit_feature_scales, aggregate, thresholds

ROOT=Path(__file__).resolve().parents[1]
TASKS=('ulb','banksim','s_ffsd','sapienza_5S','sapienza_15S','sapienza_25S','sd_stock','sd_crypto','sd_synth')
SEEDS=(42,43,44)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(2**20):h.update(b)
    return h.hexdigest()


def sources():
    declared=__import__('os').environ.get('FINAD_CODE_COMMIT')
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit=declared
    if not commit or len(commit)!=40 or declared and commit!=declared:raise ValueError('Need exact source commit')
    files=[ROOT/'finad_kronos'/name for name in ['stable.py','stable_experiment.py']]
    files += [ROOT/'finad'/name for name in ['preprocessing.py','thresholds.py','evaluation.py']]
    return dict(git_commit=commit,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
                python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn']})


def check_inputs(data):
    golden=json.loads((ROOT/'results/zero_label_20261114/input_cache_files_sha256.json').read_text())
    for name,info in golden.items():
        p=data/name
        if p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']:raise ValueError('Changed paired input: '+name)
    return golden


def causal_components(data,raw):
    """Returns feature-only components; target label files are never decoded."""
    meta=json.loads((data/'manifest.json').read_text());width=len(meta['features'])
    x={s:np.memmap(data/f'{s}.x.f32',mode='r',dtype='float32',shape=(meta['splits'][s]['rows'],width)) for s in ['train','calibration','test']}
    pre=Preprocessor.load(raw/'preprocessor.npz')
    with np.load(raw/'indices.npz',allow_pickle=False) as z:ti,ci,test=z['train'],z['calibration'],z['test']
    np.testing.assert_array_equal(test,np.arange(len(x['test'])))
    rebuilt=Preprocessor().fit(x['train'][ti])
    for k in ('keep','missing','center','scale'):np.testing.assert_array_equal(getattr(pre,k),getattr(rebuilt,k))
    ztrain=pre.transform(x['train']);prior=fit_prior(ztrain[ti]);center=np.asarray(prior['center'])
    values=np.concatenate([ztrain,pre.transform(x['calibration']),pre.transform(x['test'])])
    if values.shape[1]!=width:raise ValueError('Unexpected transformed feature schema')
    nt,nc=len(x['train']),len(x['calibration']);starts=np.zeros(len(values),dtype=np.int64)
    if meta.get('temporal'):
        for s,offset in [('train',0),('calibration',nt),('test',nt+nc)]:
            seq=np.fromfile(data/f'{s}.seq.i32',dtype=np.int32)
            boundaries=np.r_[0,np.flatnonzero(np.diff(seq))+1,len(seq)]
            for left,right in zip(boundaries[:-1],boundaries[1:]):starts[offset+left:offset+right]=offset+left
    components={}
    with np.load(raw/'forecasts.npz',allow_pickle=False) as forecasts:
        for phase,positions in [('calibration',nt+ci),('test',nt+nc+test)]:
            std=np.empty((len(positions),width),dtype=np.float64)
            counts=np.minimum(32,positions-starts[positions])
            for begin in range(0,len(positions),256):
                t=positions[begin:begin+256];idx=t[:,None]-np.arange(32,0,-1);valid=idx>=starts[t,None]
                past=values[np.maximum(idx,0)].astype('float64');past[~valid]=0
                std[begin:begin+len(t)]=past.std(axis=1)
            last=values[np.maximum(positions-1,0)].copy().astype('float64');last[counts==0]=center
            components[phase]=dict(observed=values[positions],predicted=forecasts[phase],history_std=std,counts=counts,last=last)
    return components,prior,meta


def score_all(a):
    a.output.mkdir(parents=True,exist_ok=True)
    original=json.loads((a.original_runs/'run_manifest.json').read_text())
    if json.loads((a.original_runs/'verification.json').read_text())['runs']!=27:raise ValueError('Need verified original cohort')
    original_records=json.loads((a.original_runs/'records.json').read_text())
    expected={(d,s) for d in TASKS for s in SEEDS}
    assert {(r['dataset'],r['seed']) for r in original_records}==expected
    manifest={**sources(),tasks=list(TASKS),seeds=list(SEEDS),variants=list(VARIANTS),
              protocol_sha256=sha(ROOT/'docs/kronos_stable_protocol_CN.md'),
              original_scoring_commit=original['git_commit'],inference_commit=original['inference_source_commit'],
              assessment='Retrospective development on previously inspected test splits; not independent validation',
              reused_frozen_forecasts=True,new_backbone_inference=False,primary_variant='nc',
              label_policy='All 162 score files and calibration rules completed before separate target decoding stage'}
    p=a.output/'run_manifest.json'
    if p.exists() and json.loads(p.read_text())!=manifest:raise ValueError('Source/environment/config changed; choose new output')
    write(p,manifest);write(a.output/'input_files_sha256.json',check_inputs(a.data))
    records=[]
    for d in TASKS:
        for seed in SEEDS:
            started=time.monotonic();raw=a.raw_runs/d/str(seed);old=a.original_runs/d/str(seed);run=a.output/d/str(seed)
            prior_record=next(r for r in original_records if (r['dataset'],r['seed'])==(d,seed))
            if sha(raw/'forecasts.npz')!=prior_record['checks']['forecasts_sha256']:raise ValueError('Frozen forecast changed')
            with np.load(raw/'indices.npz',allow_pickle=False) as z:
                meta=json.loads((a.data/d/'manifest.json').read_text())
                for split,k,offset in [('train','train',0),('calibration','calibration',100)]:
                    n=meta['splits'][split]['rows'];size=min(8192 if split=='train' else 100000,n)
                    np.testing.assert_array_equal(z[k],np.sort(np.random.default_rng(seed+offset).choice(n,size,replace=False)))
            comps,prior,meta=causal_components(a.data/d,raw);write(run/'training_prior.json',prior)
            for variant in VARIANTS:
                out=run/variant;out.mkdir(parents=True,exist_ok=True)
                ce=errors(**comps['calibration'],prior=prior,variant=variant)
                fitted=json.loads((old/'feature_pot.json').read_text()) if variant=='legacy' else fit_feature_scales(ce,meta['features'])
                cal=aggregate(ce,fitted);del ce
                te=errors(**comps['test'],prior=prior,variant=variant);test=aggregate(te,fitted);del te
                if variant=='legacy':
                    with np.load(old/'scores.npz',allow_pickle=False) as saved:
                        np.testing.assert_allclose(cal,saved['calibration'],rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(test,saved['test'],rtol=1e-12,atol=1e-12)
                th=thresholds(cal)
                np.savez_compressed(out/'scores.npz',calibration=cal,test=test)
                write(out/'feature_pot.json',fitted);write(out/'thresholds.json',th)
                cold={phase:dict(scored=len(c['counts']),cold_start=int(np.sum(c['counts']<32)),no_history=int(np.sum(c['counts']==0))) for phase,c in comps.items()}
                recipe=dict(dataset=d,seed=seed,variant=variant,method=METHODS[variant],
                            training_prior_sha256=sha(run/'training_prior.json'),
                            frozen_forecasts_sha256=sha(raw/'forecasts.npz'),preprocessor_sha256=sha(raw/'preprocessor.npz'),
                            indices_sha256=sha(raw/'indices.npz'),scores_sha256=sha(out/'scores.npz'),
                            feature_pot_sha256=sha(out/'feature_pot.json'),thresholds_sha256=sha(out/'thresholds.json'),
                            active_dimensions=len(fitted['active_indices']),total_dimensions=len(meta['features']),
                            cold_start_counts=cold,seconds_in_dataset_seed=time.monotonic()-started,
                            new_inference=False,backbone_used=variant not in ('prior','persist'))
                write(out/'recipe.json',recipe);records.append(recipe)
            print('SCORED_FIXED_SIX_VARIANTS',d,seed,f'{len(records)}/162',flush=True)
            del comps
    assert len(records)==162
    write(a.output/'score_records.json',records);write(a.output/'scoring_complete.json',dict(finished=162,failed=0,expected=162))


def evaluate_all(a):
    assert json.loads((a.output/'scoring_complete.json').read_text())==dict(finished=162,failed=0,expected=162)
    records=json.loads((a.output/'score_records.json').read_text())
    assert {(r['dataset'],r['seed'],r['variant']) for r in records}=={(d,s,v) for d in TASKS for s in SEEDS for v in VARIANTS}
    evaluated=[]
    for r in records:
        run=a.output/r['dataset']/str(r['seed'])/r['variant']
        for name,key in [('scores.npz','scores_sha256'),('thresholds.json','thresholds_sha256'),('feature_pot.json','feature_pot_sha256')]:assert sha(run/name)==r[key]
        with np.load(run/'scores.npz',allow_pickle=False) as z:score=z['test']
        y=np.fromfile(a.data/r['dataset']/'test.y.i8',dtype=np.int8)
        th=json.loads((run/'thresholds.json').read_text())
        result={**r,'metrics':metrics(y,score,th)}
        write(run/'result.json',result);evaluated.append(result)
    write(a.output/'records.json',evaluated);write(a.output/'complete.json',dict(finished=162,failed=0,expected=162))
    print('EVALUATED_ALL_FIXED_CONFIGURATIONS',len(evaluated),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-runs',type=Path,required=True);p.add_argument('--original-runs',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['score','evaluate'],default='score');a=p.parse_args()
    score_all(a) if a.stage=='score' else evaluate_all(a)


if __name__=='__main__':main()
