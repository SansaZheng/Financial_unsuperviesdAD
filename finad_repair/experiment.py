"""Prespecified repair factorials; target labels opened only after score gates."""
import argparse
import gc
import json
from pathlib import Path
import platform
import time
import shutil
import copy
import numpy as np
import scipy
import torch
import sklearn
import pywt
from threadpoolctl import threadpool_limits
from finad.preprocessing import Preprocessor
from finad_recent.data import ROOT, PHASES, TEMPORAL, GRAPH, sha, write, verify_inputs, transaction_keys
from finad_recent.experiment import load_tabular, finish
from finad_recent.huge import HUGE, bidirected_union
from finad_recent.himvh import HIMVHZero, hyperedges, discrepancies
from finad_recent.sdfmm import SDFMMZero, sequence_starts
from finad_zero.graph_data import load_phase
from finad_zero.experiment import code_commit
from .roles import RoleNormalizer, score_without_self
from .himvh import HIMVHRepair
from .sdfmm import SDRepair

VARIANTS = dict(huge=['legacy', 'role', 'role_long', 'role_noself'],
                himvh=['legacy', 'f00', 'c', 'm', 'cm', 'm_flat', 'cm_flat'],
                sdfmm=['legacy', 'f00', 'e', 'g', 'ge', 'ge_nocontrast'])
PRIMARY = dict(huge='role', himvh='cm', sdfmm='ge')


def planned(families=('huge', 'himvh', 'sdfmm'), seeds=(42,43,44)):
    return [(family, task, variant, seed) for family in families
            for task in (TEMPORAL if family == 'sdfmm' else GRAPH)
            for seed in seeds for variant in VARIANTS[family]]


def raw_values(data, meta):
    return {s: np.memmap(data/f'{s}.x.f32', mode='r', dtype='float32',
                        shape=(meta['splits'][s]['rows'],len(meta['features']))) for s in PHASES}


def reload_net(model, loaded, x, device):
    # CUDA scatter reductions are order-dependent even for identical weights.
    # Check every persisted tensor bitwise, then compare CPU evaluation exactly;
    # do not weaken tolerances or alter the training/scoring implementation.
    model.net.eval(); loaded.net.eval()
    state = model.net.state_dict(); restored = loaded.net.state_dict()
    assert state.keys() == restored.keys()
    for name in state:
        assert torch.equal(state[name].detach().cpu(), restored[name].detach().cpu()), name
    width = x.shape[1]; probe = x[:min(64, len(x))]
    views = [hyperedges(np.arange(len(probe))//8) for _ in range(4)]
    enhanced = torch.as_tensor(discrepancies(probe, views))
    edges = [torch.as_tensor(e) for e in views]
    original_cpu = copy.deepcopy(model.net).cpu()
    restored_cpu = copy.deepcopy(loaded.net).cpu()
    with torch.no_grad(): a = original_cpu(enhanced, edges); b = restored_cpu(enhanced, edges)
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    return float((a-b).abs().max())


def graph_run(args, task, variant, seed, out, cache):
    data = args.graphs/task
    phases = {s: load_phase(data,s) for s in PHASES}
    roles = {s: v[0][0,:,0]>.5 for s,v in phases.items()}
    common = Preprocessor().fit(phases['train'][0].reshape(-1,10)); common.save(out/'preprocessor.npz')
    flat = {s: common.transform(v[0].reshape(-1,10)).reshape(v[0].shape).transpose(1,0,2).reshape(v[0].shape[1],-1) for s,v in phases.items()}
    edges = {s: bidirected_union(v[1],v[0].shape[1]) for s,v in phases.items()}
    eligible = np.flatnonzero(roles['train'])
    ti = np.sort(np.random.default_rng(seed).choice(eligible,min(8192,len(eligible)),replace=False))
    np.savez(out/'indices.npz',train=ti,calibration=np.flatnonzero(roles['calibration']),test=np.flatnonzero(roles['test']))
    if variant != 'legacy':
        normalizer = RoleNormalizer().fit(phases['train'][0],roles['train']); normalizer.save(out)
        again = RoleNormalizer.load(out)
        flat = {s:normalizer.transform(v[0],roles[s]) for s,v in phases.items()}
        np.testing.assert_array_equal(flat['test'],again.transform(phases['test'][0],roles['test']))
    if variant == 'role_noself':
        parent = cache['role']; shutil.copyfile(parent/'model.pt',out/'model.pt')
        model = HUGE.load(out/'model.pt',args.device)
    else:
        model = HUGE(flat['train'].shape[1],seed,args.device,1200 if variant=='role_long' else 300).fit(flat['train'],edges['train'])
        model.save(out/'model.pt')
    loaded = HUGE.load(out/'model.pt',args.device)
    a = model.score(flat['calibration'],edges['calibration']); b = loaded.score(flat['calibration'],edges['calibration'])
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
    scores = {s:model.score(flat[s],edges[s]) for s in ['calibration','test']}
    if variant == 'role_noself': scores={s:score_without_self(model,flat[s],edges[s]) for s in scores}
    cs, ts = scores['calibration'][roles['calibration']],scores['test'][roles['test']]
    details = dict(checkpoint_max_abs=float(np.max(np.abs(a-b))),test_target_sha256=sha(data/'test.y.npy'),
                   method_input='common40' if variant=='legacy' else 'role-canonical16; role-specific train-only scaling',
                   source_manifest=json.loads((data/'manifest.json').read_text()))
    model.info.update(variant=variant,role_alignment=variant!='legacy',include_score_self=variant!='role_noself',
                      fit_reused=variant=='role_noself')
    return finish(out,cs,ts,lambda:np.load(data/'test.y.npy',allow_pickle=False)[roles['test']]),model.info,details


def tabular_run(args, task, family, variant, seed, out, cache):
    data=args.data/task; meta,x,ti,ci=load_tabular(data,out,seed)
    raw=raw_values(data,meta);pre=Preprocessor.load(out/'preprocessor.npz')
    np.testing.assert_array_equal(pre.transform(raw['test'][:64]),x['test'][:64])
    probe_error=0.;details={}
    if family=='sdfmm':
        starts={s:sequence_starts(np.fromfile(data/f'{s}.seq.i32',dtype=np.int32)) for s in PHASES}
        if variant=='legacy':
            model=SDFMMZero(x['train'].shape[1],seed,args.device,400).fit(x['train'],starts['train'],ti)
            model.save(out/'model.pt');loaded=SDFMMZero.load(out/'model.pt',args.device)
            probe=np.arange(min(64,len(x['test'])))
            a=model.score(x['test'],starts['test'],probe);b=loaded.score(x['test'],starts['test'],probe)
            cs=model.score(x['calibration'],starts['calibration'],ci)
            ts=model.score(x['test'],starts['test'],np.arange(len(x['test'])))
        else:
            model=SDRepair(x['train'].shape[1],task,seed,args.device,endpoint=variant in ['e','ge','ge_nocontrast'],
                           generator=variant in ['g','ge','ge_nocontrast'],contrastive=0. if variant=='ge_nocontrast' else .5)
            model.fit(raw['train'],starts['train'],ti,pre);model.save(out/'model.pt');loaded=SDRepair.load(out/'model.pt',args.device)
            probe=np.arange(min(64,len(x['test'])))
            a=model.score_raw(raw['test'],starts['test'],probe,pre);b=loaded.score_raw(raw['test'],starts['test'],probe,pre)
            cs=model.score_raw(raw['calibration'],starts['calibration'],ci,pre)
            ts=model.score_raw(raw['test'],starts['test'],np.arange(len(x['test'])),pre)
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6);probe_error=float(np.max(np.abs(a-b)))
    else:
        native={s:transaction_keys(task,s)[0] for s in PHASES}
        selected=transaction_keys(task,'train',ti)[0]
        if variant=='legacy':
            model=HIMVHZero(x['train'].shape[1],4,seed,args.device,200).fit(x['train'][ti],selected)
            model.save(out/'model.pt');loaded=HIMVHZero.load(out/'model.pt',args.device)
            probe_error=reload_net(model,loaded,x['train'][ti],args.device)
            cs=model.score(x['calibration'],native['calibration'])[ci];ts=model.score(x['test'],native['test'])
        else:
            c=variant in ['c','cm','cm_flat'];m=variant in ['m','cm','m_flat','cm_flat']
            if variant.endswith('_flat'):
                parent=cache[variant.removesuffix('_flat')];shutil.copyfile(parent/'model.pt',out/'model.pt')
                model=HIMVHRepair.load(out/'model.pt',args.device)
                with np.load(parent/'components.npz',allow_pickle=False) as z:calcomp,testcomp=z['calibration'],z['test']
            else:
                model=HIMVHRepair(x['train'].shape[1],task,seed,args.device,c,m)
                trainx=x['train'] if c else x['train'][ti]
                model.fit(trainx,native['train'] if c else selected,ti if c else np.arange(len(ti)),None if c else ti)
                model.save(out/'model.pt')
                calcomp=model.components(x['calibration'],native['calibration'])[ci]
                testcomp=model.components(x['test'],native['test'])
            np.savez_compressed(out/'components.npz',calibration=calcomp,test=testcomp)
            loaded=HIMVHRepair.load(out/'model.pt',args.device)
            probe_error=reload_net(model,loaded,x['train'][ti],args.device)
            cs=model.score_components(calcomp,flat=variant.endswith('_flat'))
            ts=model.score_components(testcomp,flat=variant.endswith('_flat'))
            model.info.update(flat_score=variant.endswith('_flat'),fit_reused=variant.endswith('_flat'))
        details['hyperedges']={s:[len(e) for e in native[s]] for s in PHASES}
    model.info.update(variant=variant)
    details.update(checkpoint_max_abs=probe_error,test_target_sha256=sha(data/'test.y.i8'),source_manifest=meta)
    return finish(out,cs,ts,lambda:np.fromfile(data/'test.y.i8',dtype=np.int8)),model.info,details


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=list(VARIANTS),required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if a.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    inputs=verify_inputs(a.data,a.graphs);write(a.output/'input_files_sha256.json',inputs)
    legacy=json.loads((ROOT/'results/recent_zero_20261117/run_manifest.json').read_text())
    for name,h in legacy['source_sha256'].items():
        if name.startswith('finad_recent/') or name.startswith('finad/'):
            assert sha(ROOT/name)==h,('Immutable baseline code changed',name)
    files=list((ROOT/'finad_repair').glob('*.py'))+list((ROOT/'finad_recent').glob('*.py'))+[ROOT/'finad'/n for n in ['preprocessing.py','thresholds.py','evaluation.py']]
    manifest=dict(git_commit=code_commit(),source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)},
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
                  pywavelets=pywt.__version__,family=a.family,seeds=a.seeds,variants=VARIANTS[a.family],primary=PRIMARY[a.family])
    write(a.output/'run_manifest.json',manifest);jobs=planned([a.family],a.seeds);records=[];cache={}
    for i,(family,task,variant,seed) in enumerate(jobs,1):
        out=a.output/('graph' if family=='huge' else 'tabular')/task/family/variant/str(seed);out.mkdir(parents=True,exist_ok=True)
        print(f'RUN {i}/{len(jobs)} {family}/{task}/{variant}/{seed}',flush=True);started=time.monotonic()
        if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        with threadpool_limits(limits=4):
            result,info,checks=graph_run(a,task,variant,seed,out,cache) if family=='huge' else tabular_run(a,task,family,variant,seed,out,cache)
        record=dict(track='graph' if family=='huge' else 'tabular',dataset=task,family=family,variant=variant,method=f'{family}_repair_{variant}',seed=seed,
                    seconds=time.monotonic()-started,metrics=result,model_info=info,checks=checks,checkpoint_sha256=sha(out/'model.pt'),score_sha256=sha(out/'scores.npz'),
                    peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None)
        write(out/'result.json',record);records.append(record);cache[variant]=out
        write(a.output/'progress.json',dict(finished=len(records),expected=len(jobs),failed=0))
        print(f'DONE {i}/{len(jobs)} {family}/{task}/{variant}/{seed} seconds={record["seconds"]:.1f}',flush=True)
        gc.collect()
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
    write(a.output/'records.json',records);write(a.output/'complete.json',dict(finished=len(records),expected=len(jobs),failed=0))


if __name__=='__main__':main()
