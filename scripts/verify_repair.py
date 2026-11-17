"""Independent score/threshold replay and sparse-ego CPU checkpoint probes."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc
from finad.preprocessing import Preprocessor
from finad_recent.data import sha,write,PHASES,transaction_keys
from finad_zero.graph_data import load_phase
from finad_recent.huge import HUGE,bidirected_union
from finad_recent.himvh import HIMVHZero
from finad_recent.sdfmm import SDFMMZero,sequence_starts
from finad_repair.roles import RoleNormalizer,score_without_self
from finad_repair.himvh import HIMVHRepair
from finad_repair.sdfmm import SDRepair
from finad_repair.hypergraph import NativeContext,block_forward,assignments,masked
from finad_repair.experiment import planned,PRIMARY


def close(a,b,tol=1e-10):
    if a is None or b is None: assert a is None and b is None
    else: np.testing.assert_allclose(a,b,rtol=tol,atol=tol)


def probe_him(model,x,views,rows,legacy):
    ctx=NativeContext(x,views);plan=ctx.plan(rows)
    needed=np.unique(np.concatenate([b[0][0] for b in plan]));model.net.eval()
    groups=[np.arange(x.shape[1])] if legacy else model.groups
    result=np.zeros((len(rows),len(groups)))
    consistent=False if legacy else model.consistent
    base=assignments(ctx.ids,model.seed+991,len(groups))
    with torch.no_grad():
        for turn in range(len(groups) if consistent else 1):
            fields=(base+turn)%len(groups)
            value=masked(x,fields,groups) if consistent else x
            enhanced=ctx.enhanced(value,needed)
            pred=block_forward(model.net,enhanced,plan,rows,'cpu',needed).numpy()
            error=(pred-x[rows])**2
            for g,cols in enumerate(groups):
                selected=fields[rows]==g if consistent else np.ones(len(rows),bool)
                result[selected,g]=error[selected][:,cols].mean(1)
    return result[:,0] if legacy else result


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True,help='folder with huge/himvh/sdfmm output folders')
    p.add_argument('--old-runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];records=[];manifests={};seen=set();verified=[];dependencies={};shards={};torch.set_num_threads(4)
    for family in ['huge','himvh','sdfmm']:
        job=a.jobs/family;run=job/'results';manifest=json.loads((run/'run_manifest.json').read_text());manifests[family]=manifest
        expected=planned([family]);assert json.loads((run/'complete.json').read_text())==dict(finished=len(expected),expected=len(expected),failed=0)
        for name,h in manifest['source_sha256'].items():
            data=subprocess.check_output(['git','-C',str(root),'show',manifest['git_commit']+':'+name]);import hashlib
            assert hashlib.sha256(data).hexdigest()==h
        for name in ['scripts/prepare_repo_data.py','finad_zero/graph_data.py',
                     'data/raw/banksim/banksim.zip','data/raw/s_ffsd/data/S-FFSD.zip']:
            data=subprocess.check_output(['git','-C',str(root),'show',manifest['git_commit']+':'+name])
            h=hashlib.sha256(data).hexdigest();assert sha(root/name)==h
            dependencies[name]=dict(sha256=h,bytes=len(data))
        inputs=json.loads((run/'input_files_sha256.json').read_text());assert len(inputs)==101
        assert inputs==json.loads((root/'results/recent_zero_20261117/input_files_sha256.json').read_text())
        for name,record in inputs.items():
            path=job/'inputs'/name
            # Input keys use tabular/... or graphs/..., same staged layout.
            assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
        family_records=json.loads((run/'records.json').read_text());assert len(family_records)==len(expected)
        if manifest.get('composed_from_independent_seed_shards'):
            parts=json.loads((run/'shard_provenance.json').read_text());shards[family]=parts
            assert family in ['sdfmm','himvh'] and len(parts)==3
            assert {r['seed'] for r in parts}=={42,43,44}
            for part in parts:
                original=part['manifest'];seed=part['seed']
                assert original['seeds']==[seed] and original['git_commit']==manifest['git_commit']
                assert {k:v for k,v in original.items() if k!='seeds'}=={k:v for k,v in manifest.items() if k not in ['seeds','composed_from_independent_seed_shards']}
                n=len(planned([family],[seed]));values=[r for r in family_records if r['seed']==seed];assert len(values)==n
                def digest_json(value):return hashlib.sha256((json.dumps(value,indent=2)+'\n').encode()).hexdigest()
                assert digest_json(original)==part['manifest_sha256']
                assert digest_json(values)==part['records_sha256']
                assert digest_json(dict(finished=n,expected=n,failed=0))==part['complete_sha256']
        contexts={};views_cache={}
        for r in family_records:
            track,task,v,seed=r['track'],r['dataset'],r['variant'],r['seed'];key=(family,task,v,seed)
            assert key not in seen;seen.add(key)
            info=r['model_info'];assert info['variant']==v
            expected_steps=(1200 if v=='role_long' else 300) if family=='huge' else ((200 if v=='legacy' else 800) if family=='himvh' else 400)
            assert info['steps']==expected_steps
            reused=v=='role_noself' or v.endswith('_flat')
            assert bool(info.get('fit_reused',False))==reused
            folder=run/track/task/family/v/str(seed);old=a.old_runs/track/task/'ae'/str(seed)
            assert sha(folder/'model.pt')==r['checkpoint_sha256'] and sha(folder/'scores.npz')==r['score_sha256']
            gate=json.loads((folder/'label_access_gate.json').read_text())
            for field,name in [('checkpoint_sha256','model.pt'),('scores_sha256','scores.npz'),('thresholds_sha256','thresholds.json')]:assert gate[field]==sha(folder/name)
            with np.load(folder/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
            with np.load(old/'indices.npz',allow_pickle=False) as z:
                for k in z.files:np.testing.assert_array_equal(idx[k],z[k])
            with np.load(folder/'scores.npz',allow_pickle=False) as z:cal,score=z['calibration'],z['test']
            group=(task,seed)
            if group not in contexts:
                if track=='graph':
                    graph=job/'inputs/graphs'/task;raw={s:load_phase(graph,s) for s in PHASES}
                    roles={s:t[0][0,:,0]>.5 for s,t in raw.items()}
                    pre=Preprocessor().fit(raw['train'][0].reshape(-1,10));fitted_input=raw['train'][0].reshape(-1,10)
                    context=dict(raw=raw,roles=roles,pre=pre,source=graph)
                else:
                    data=job/'inputs/tabular'/task;meta=json.loads((data/'manifest.json').read_text());width=len(meta['features'])
                    raw={s:np.memmap(data/f'{s}.x.f32',mode='r',dtype='float32',shape=(meta['splits'][s]['rows'],width)) for s in PHASES}
                    ti=np.sort(np.random.default_rng(seed).choice(len(raw['train']),min(8192,len(raw['train'])),replace=False))
                    ci=np.sort(np.random.default_rng(seed+100).choice(len(raw['calibration']),min(100000,len(raw['calibration'])),replace=False))
                    np.testing.assert_array_equal(ti,idx['train']);np.testing.assert_array_equal(ci,idx['calibration'])
                    pre=Preprocessor().fit(raw['train'][ti]);context=dict(raw=raw,pre=pre,source=data,meta=meta)
                    fitted_input=raw['train'][ti]
                previous=Preprocessor.load(old/'preprocessor.npz')
                for k in ['center','scale']:np.testing.assert_allclose(getattr(previous,k),getattr(pre,k),rtol=1e-14,atol=1e-14)
                contexts.clear();contexts[group]=context
            ctx=contexts[group];raw=ctx['raw'];pre=Preprocessor.load(folder/'preprocessor.npz')
            for s in PHASES:
                observation=raw[s][0].reshape(-1,10) if track=='graph' else raw[s]
                previous=Preprocessor.load(old/'preprocessor.npz')
                np.testing.assert_array_equal(pre.transform(observation),previous.transform(observation))
            model_probe=[]
            if track=='graph':
                roles=ctx['roles'];label=ctx['source']/'test.y.npy';y=np.load(label,allow_pickle=False)[roles['test']]
                edges={s:bidirected_union(raw[s][1],raw[s][0].shape[1]) for s in PHASES}
                if v=='legacy':z={s:pre.transform(raw[s][0].reshape(-1,10)).reshape(raw[s][0].shape).transpose(1,0,2).reshape(raw[s][0].shape[1],-1) for s in PHASES}
                else:
                    norm=RoleNormalizer.load(folder);fit=RoleNormalizer().fit(raw['train'][0],roles['train']);z={}
                    for s in PHASES:
                        z[s]=norm.transform(raw[s][0],roles[s]);np.testing.assert_array_equal(z[s],fit.transform(raw[s][0],roles[s]))
                m=HUGE.load(folder/'model.pt','cpu')
                for phase,saved in [('calibration',cal),('test',score)]:
                    pred=(score_without_self(m,z[phase],edges[phase]) if v=='role_noself' else m.score(z[phase],edges[phase]))[roles[phase]]
                    np.testing.assert_allclose(pred,saved,rtol=5e-4,atol=2e-5);model_probe.append(float(np.max(np.abs(pred-saved))))
            else:
                label=ctx['source']/'test.y.i8';y=np.fromfile(label,dtype=np.int8)
                np.testing.assert_array_equal(idx['test'],np.arange(len(y)))
                rows=np.arange(min(32,len(y)))
                if family=='sdfmm':
                    starts=sequence_starts(np.fromfile(ctx['source']/'test.seq.i32',dtype=np.int32))
                    if v=='legacy':m=SDFMMZero.load(folder/'model.pt');pred=m.score(pre.transform(raw['test']),starts,rows)
                    else:m=SDRepair.load(folder/'model.pt');pred=m.score_raw(raw['test'],starts,rows,pre)
                else:
                    if task not in views_cache:views_cache[task]=transaction_keys(task,'test')[0]
                    x=pre.transform(raw['test']);view=views_cache[task]
                    m=HIMVHZero.load(folder/'model.pt') if v=='legacy' else HIMVHRepair.load(folder/'model.pt')
                    comp=probe_him(m,x,view,rows,v=='legacy')
                    pred=comp if v=='legacy' else m.score_components(comp,flat=v.endswith('_flat'))
                    if v!='legacy':
                        with np.load(folder/'components.npz',allow_pickle=False) as z:
                            np.testing.assert_allclose(comp,z['test'][rows],rtol=5e-4,atol=2e-5)
                            for phase,saved in [('calibration',cal),('test',score)]:close(m.score_components(z[phase],flat=v.endswith('_flat')),saved)
                np.testing.assert_allclose(pred,score[rows],rtol=5e-4,atol=2e-5);model_probe.append(float(np.max(np.abs(pred-score[rows]))))
            assert sha(label)==r['checks']['test_target_sha256']
            assert len(cal)==len(idx['calibration']) and len(score)==len(y)
            assert np.isfinite(cal).all() and np.isfinite(score).all()
            metrics=r['metrics'];oldmetric=json.loads((old/'result.json').read_text())['metrics']
            for k in ['evaluated','unknown_excluded','positives','normal','prevalence']:close(metrics[k],oldmetric[k])
            known=y!=-1;target=y[known];s=score[known];p_,recall_,_=precision_recall_curve(target,s)
            for k,value in [('average_precision',average_precision_score(target,s)),('roc_auc',roc_auc_score(target,s)),('pr_auc_trapezoid',auc(recall_,p_))]:close(metrics[k],value)
            thresholds=json.loads((folder/'thresholds.json').read_text());u=np.quantile(cal,.85);peaks=cal[cal>u]-u
            for name,th in thresholds.items():
                if th['status']!='ok':assert metrics['operating_points'][name] is None and th.get('reason');continue
                if name=='quantile':expected=float(np.quantile(cal,.99))
                else:
                    assert len(peaks)>=20 and np.ptp(peaks)>0
                    shape,_,scale=genpareto.fit(peaks,floc=0);close(shape,th['shape']);close(scale,th['scale'])
                    expected=u+genpareto.isf(.01*len(cal)/len(peaks),shape,loc=0,scale=scale)
                close(th['threshold'],expected)
                alarm=s>=th['threshold'];tp=int(np.sum((target==1)&alarm));fp=int(np.sum((target==0)&alarm));fn=int(np.sum((target==1)&~alarm));tn=int(np.sum((target==0)&~alarm))
                op=metrics['operating_points'][name]
                for k,value in dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
                                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,fpr=fp/(fp+tn),alarm_fraction=float(alarm.mean())).items():close(op[k],value)
            if v=='role_noself' or v.endswith('_flat'):
                parent=folder.parents[1]/('role' if v=='role_noself' else v.removesuffix('_flat'))/str(seed)
                assert sha(parent/'model.pt')==sha(folder/'model.pt')
            verified.append(dict(family=family,dataset=task,variant=v,seed=seed,checkpoint_cpu_probe_max_abs=max(model_probe),shared_float32_observations_bitwise_equal=True))
            records.append(r)
    assert seen==set(planned()) and len(records)==174
    assert sum(not r['model_info'].get('fit_reused',False) for r in records)==156
    assert len({m['git_commit'] for m in manifests.values()})==1
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'records.json',records);write(a.output/'run_manifests.json',manifests)
    write(a.output/'shard_provenance.json',shards)
    write(a.output/'verification.json',dict(status='passed',runs=174,operating_records=348,actual_fits=156,input_files_per_family=101,
          checkpoint_cpu_probe_tolerance=dict(rtol=5e-4,atol=2e-5),verified=verified,native_key_dependencies=dependencies,
          checks=['frozen_source','input_SHA256','checkpoint_score_gate_SHA256','same_target_sampling_and_cohorts','shared_float32_observations',
                  'role_train_normalizers','CPU_checkpoint_probes','exact_ego_forward_gradient_unit_tests','field_score_aggregation','AP_ROC_PR','refit_POT_q99','confusion_counts','derived_checkpoint_identity']))
    print('VERIFIED174/174 configurations and348 operating points; no new fitting')


if __name__=='__main__':main()
