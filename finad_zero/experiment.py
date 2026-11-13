"""Fixed-budget experiments; fit/calibration never open training/calibration labels.

Checkpoint + calibration/test scores are written before test labels are opened.
Original benchmark files are not modified. Private scores permit independent audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits
from finad.preprocessing import Preprocessor
from finad.classical import ClassicalDetector
from finad.neural import NeuralDetector
from finad.thresholds import calibrate
from finad.evaluation import metrics
from .cae_if import CAEIF
from .audit_hcl import AuditHCL
from .graph_data import load_phase, induce
from .ebgad import fetch_upstream, score_graph

TASKS=('ulb','banksim','s_ffsd','sapienza_5S','sapienza_15S','sapienza_25S','sd_stock','sd_crypto','sd_synth')
TABULAR=('cae_if_core','iforest','ae','pca')
GRAPH=TABULAR+('ebgad_release_zero','audit_hcl_zero_adapted')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(2**20): h.update(chunk)
    return h.hexdigest()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def source_fingerprint():
    root=Path(__file__).resolve().parents[1]
    files=list((root/'finad_zero').glob('*.py'))+[root/'finad'/f for f in
        ('classical.py','neural.py','preprocessing.py','thresholds.py','evaluation.py')]+[root/'scripts/prepare_repo_data.py']
    return {str(p.relative_to(root)):sha(p) for p in sorted(files)}


def make_model(method,seed,device,steps):
    if method=='cae_if_core': return CAEIF(seed,device,steps)
    if method=='ae': return NeuralDetector('ae',seed,device,steps=steps)
    return ClassicalDetector(method,seed)


def score_chunks(model,x):
    return np.concatenate([np.asarray(model.score(x[i:i+4096]),dtype=np.float64) for i in range(0,len(x),4096)])


def tabular_run(data,out,method,seed,device,steps):
    meta=json.loads((data/'manifest.json').read_text()); p=len(meta['features'])
    x={s:np.memmap(data/f'{s}.x.f32',mode='r',dtype='float32',shape=(meta['splits'][s]['rows'],p))
       for s in ('train','calibration','test')}
    indices=np.sort(np.random.default_rng(seed).choice(len(x['train']),min(8192,len(x['train'])),replace=False))
    ci=np.sort(np.random.default_rng(seed+100).choice(len(x['calibration']),min(100000,len(x['calibration'])),replace=False))
    pre=Preprocessor().fit(x['train'][indices]); z=pre.transform(x['train'][indices]); pre.save(out/'preprocessor.npz')
    np.savez(out/'indices.npz',train=indices,calibration=ci)
    model=make_model(method,seed,device,steps).fit(z)
    path=out/'model'; model.save(path)
    loaded=type(model).load(path,device=device) if method in ('cae_if_core','ae') else type(model).load(path)
    probe=pre.transform(x['test'][:min(257,len(x['test']))])
    a,b=model.score(probe),loaded.score(probe)
    if not np.allclose(a,b,rtol=1e-5,atol=1e-6): raise AssertionError('Checkpoint reload mismatch')
    cs=score_chunks(model,pre.transform(x['calibration'][ci])); ts=score_chunks(model,pre.transform(x['test']))
    np.savez_compressed(out/'scores.npz',calibration=cs,test=ts)
    thresholds=calibrate(cs); write(out/'thresholds.json',thresholds)
    # First and only target read in this run. No training or calibration label read.
    y=np.fromfile(data/'test.y.i8',dtype=np.int8)
    return metrics(y,ts,thresholds),model.info,dict(checkpoint_max_abs=float(np.max(np.abs(a-b))),
        source_manifest=meta,train_indices_sha256=sha(out/'indices.npz'),test_target_sha256=sha(data/'test.y.i8'))


def graph_run(data,out,method,seed,device,steps,audit_steps,cache,upstream):
    phases={s:load_phase(data,s) for s in ('train','calibration','test')}
    raw,_,_=phases['train']; role={s:p[0][0,:,0]>.5 for s,p in phases.items()}
    # One train-only scaler over snapshot observations shared by every graph method.
    train_flat=raw.reshape(-1,raw.shape[-1])
    pre=Preprocessor().fit(train_flat); pre.save(out/'preprocessor.npz')
    phases={s:(pre.transform(x.reshape(-1,x.shape[-1])).reshape(4,x.shape[1],-1),e,a)
            for s,(x,e,a) in phases.items()}
    flat={s:x.transpose(1,0,2).reshape(x.shape[1],-1) for s,(x,_,_) in phases.items()}
    eligible=np.flatnonzero(role['train'])
    indices=np.sort(np.random.default_rng(seed).choice(eligible,min(8192,len(eligible)),replace=False))
    np.savez(out/'indices.npz',train=indices,calibration=np.flatnonzero(role['calibration']),test=np.flatnonzero(role['test']))
    details={}
    if method=='audit_hcl_zero_adapted':
        model=AuditHCL(seed,device,audit_steps).fit(*phases['train'],role['train']); model.save(out/'model.pt')
        loaded=AuditHCL.load(out/'model.pt',device)
        scores={s:model.score(*phases[s],role[s]) for s in ('calibration','test')}
        again=loaded.score(*phases['calibration'],role['calibration'])
        if not np.allclose(scores['calibration'],again,rtol=1e-5,atol=1e-6): raise AssertionError('Audit reload mismatch')
        details['checkpoint_max_abs']=float(np.max(np.abs(scores['calibration']-again))); info=model.info
    elif method=='ebgad_release_zero':
        fetch_upstream(upstream); scores={}; info={}
        for s in ('calibration','test'):
            # Content-addressed cache prevents upstream's node-count-only key from
            # accidentally reusing a different phase's graph with the same size.
            x,e,_=phases[s]; h=hashlib.sha256()
            h.update(flat[s].tobytes())
            for edge in e: h.update(edge.tobytes())
            sub=cache/h.hexdigest(); sub.mkdir(parents=True,exist_ok=True)
            saved=sub/'selected_scores.npz'; imeta=sub/'selected_info.json'
            if saved.exists() and imeta.exists():
                with np.load(saved,allow_pickle=False) as z: scores[s]=z['score']
                info[s]=json.loads(imeta.read_text())
            else:
                scores[s],info[s]=score_graph(flat[s],e,sub/'spectrum')
                np.savez_compressed(saved,score=scores[s]); write(imeta,info[s])
        # Training-free identical procedure; seeds intentionally have no effect.
        write(out/'model_spec.json',{'upstream':info,'no_trainable_checkpoint':True})
        details['deterministic_seed_repeats']=True
    else:
        model=make_model(method,seed,device,steps).fit(flat['train'][indices]); model.save(out/'model')
        loaded=type(model).load(out/'model',device=device) if method in ('cae_if_core','ae') else type(model).load(out/'model')
        probe=flat['test'][:257]; a,b=model.score(probe),loaded.score(probe)
        if not np.allclose(a,b,rtol=1e-5,atol=1e-6): raise AssertionError('Graph checkpoint reload mismatch')
        details['checkpoint_max_abs']=float(np.max(np.abs(a-b)))
        scores={s:score_chunks(model,flat[s]) for s in ('calibration','test')};info=model.info
    cs=scores['calibration'][role['calibration']]; ts=scores['test'][role['test']]
    np.savez_compressed(out/'scores.npz',calibration=cs,test=ts)
    thresholds=calibrate(cs); write(out/'thresholds.json',thresholds)
    y=np.load(data/'test.y.npy',allow_pickle=False)[role['test']]
    details['source_manifest']=json.loads((data/'manifest.json').read_text())
    details['test_target_sha256']=sha(data/'test.y.npy')
    return metrics(y,ts,thresholds),info,details


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--graph-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=400)
    p.add_argument('--audit-steps',type=int,default=200);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--tasks',nargs='+',default=list(TASKS));p.add_argument('--tracks',nargs='+',default=['tabular','graph'])
    p.add_argument('--methods',nargs='+');args=p.parse_args()
    if args.steps<1 or args.audit_steps<1:raise ValueError('Positive fixed budgets required')
    args.output.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    env={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
         'torch':torch.__version__,'device':args.device,'source_sha256':source_fingerprint(),
         'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'steps':args.steps,'audit_steps':args.audit_steps,'seeds':args.seeds,'tasks':args.tasks,'tracks':args.tracks}
    manifest=args.output/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=env:raise ValueError('Resume config/source mismatch')
    write(manifest,env)
    jobs=[(track,task,m,seed) for track in args.tracks for task in (args.tasks if track=='tabular' else ('banksim','s_ffsd'))
          for m in (TABULAR if track=='tabular' else GRAPH) if not args.methods or m in args.methods for seed in args.seeds]
    records=[];failures=[]
    for number,(track,task,method,seed) in enumerate(jobs,1):
        out=args.output/track/task/method/str(seed);out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            records.append(json.loads((out/'result.json').read_text()));continue
        started=time.monotonic(); print(f'RUN {number}/{len(jobs)} {track}/{task}/{method}/{seed}',flush=True)
        try:
            with threadpool_limits(limits=4):
                if track=='tabular': result,info,checks=tabular_run(args.data/task,out,method,seed,args.device,args.steps)
                else: result,info,checks=graph_run(args.graph_data/task,out,method,seed,args.device,args.steps,args.audit_steps,args.cache,args.upstream)
            record=dict(track=track,dataset=task,method=method,seed=seed,seconds=time.monotonic()-started,
                        metrics=result,model_info=info,checks=checks,score_sha256=sha(out/'scores.npz'))
            write(out/'result.json',record);records.append(record)
        except Exception as error:
            failure=dict(track=track,dataset=task,method=method,seed=seed,error=str(error),traceback=traceback.format_exc())
            write(out/'failure.json',failure);failures.append(failure);print(failure['traceback'],flush=True)
        if torch.cuda.is_available():torch.cuda.empty_cache()
        write(args.output/'progress.json',dict(finished=len(records),failed=len(failures),expected=len(jobs)))
    write(args.output/'records.json',records);write(args.output/'failures.json',failures)
    write(args.output/'complete.json',dict(finished=len(records),failed=len(failures),expected=len(jobs)))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
