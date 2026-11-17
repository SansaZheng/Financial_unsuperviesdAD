"""Independent source/cohort/geometry/skill and complete metric replay."""
import argparse,gc,hashlib,json,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from finad_recent.data import ROOT,PHASES,sha,write,verify_inputs,transaction_keys
from finad.preprocessing import Preprocessor
from finad_recent.huge import HUGE,bidirected_union
from finad_zero.graph_data import load_phase,make_phase,aggregate_targets
from finad_repair.himvh import HIMVHRepair
from finad_geometry.models import HIMGeneric,groups,TASKS,HUGE_VARIANTS,FIELD_VARIANTS
from finad_geometry.experiment import views
from verify_roleweight import replay_metrics,independent_roles
from verify_repair import probe_him,checkpoint_rows,close
from finad_repair.hypergraph import NativeContext
from finad.classical import ClassicalDetector
from finad.neural import NeuralDetector
from finad_zero.audit_hcl import AuditHCL
from finad_zero.ebgad import fetch_upstream,score_graph


def tails(c,t):
    cols=[]
    for j in range(c.shape[1]):
        values,counts=np.unique(c[:,j],return_counts=True);below=np.r_[0,np.cumsum(counts)];less=below[np.searchsorted(values,t[:,j],side='left')];cols.append(-np.log((len(c)+1-less)/(len(c)+1)))
    return np.column_stack(cols)


def raw_halo(x,edges):
    a=x[edges[0]];b=x[edges[1]];coeff=np.abs(a-b)+1e-9;aa=coeff*a;bb=coeff*b
    distance=np.sqrt(np.sum((aa-bb)**2,axis=1))/np.sqrt(np.sum(aa**2,axis=1)+np.sum(bb**2,axis=1)+1e-6)
    sums=np.bincount(edges[1],weights=distance,minlength=len(x));count=np.bincount(edges[1],minlength=len(x));return np.where(count>0,sums/np.maximum(count,1),-1.)


def main():
 p=argparse.ArgumentParser();p.add_argument('--huge',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);p.add_argument('--himvh',type=Path,required=True);p.add_argument('--old-parents',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--raw-validation',type=Path,required=True);p.add_argument('--old-AE',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic();torch.set_num_threads(4)
 old=verify_inputs(a.inputs/'tabular',a.inputs/'graphs');meta=json.loads((ROOT/'docs/geometry_cardsim_archive.json').read_text());assert sha(a.raw_validation)==meta['generation']['raw_sha256'];assert len(old)==101
 for name,h in meta['files'].items():assert sha(a.inputs/name)==h['sha256'] and (a.inputs/name).stat().st_size==h['bytes']
 # Blind new world provenance; exact disjoint chronological transaction indices.
 gen=meta['generation'];assert gen['seed']==2026111707 and gen['n_payers']==1024 and gen['n_days']==90 and gen['payer_payee_factor']==1
 raw=pd.read_csv(a.raw_validation,float_precision='round_trip');times=pd.to_datetime(raw.date_time).astype('int64').to_numpy();tm=json.loads((a.inputs/'tabular/cardsim/manifest.json').read_text());rows=[]
 for k,s in enumerate(PHASES):
    idx=np.fromfile(a.inputs/'tabular/cardsim'/f'{s}.rows.i64',dtype='i8');rows.append(idx);expected=np.flatnonzero(np.searchsorted(tm['time_boundaries'],times,side='left')==k);np.testing.assert_array_equal(idx,expected)
    f=raw.iloc[idx];np.testing.assert_array_equal(np.fromfile(a.inputs/'tabular/cardsim'/f'{s}.y.i8',dtype='i1'),f.fraud.to_numpy(dtype='i1'))
    x,e,active,src,dst,ids,_=make_phase(f.payer_id,f.payee_id,times[idx],f.amount.to_numpy());saved=load_phase(a.inputs/'graphs/cardsim',s)
    np.testing.assert_allclose(x,saved[0],rtol=1e-6,atol=1e-6);np.testing.assert_array_equal(active,saved[2]);assert json.loads((a.inputs/'graphs/cardsim'/f'{s}.ids.json').read_text())==ids
    for v,w in zip(e,saved[1]):np.testing.assert_array_equal(v,w)
    np.testing.assert_array_equal(aggregate_targets(src,dst,f.fraud.to_numpy(),len(ids)),np.load(a.inputs/'graphs/cardsim'/f'{s}.y.npy',allow_pickle=False))
 np.testing.assert_array_equal(np.concatenate(rows),np.arange(len(raw)));assert times[rows[0][-1]]<times[rows[1][0]]<times[rows[2][0]]
 manifests={};records=[]
 for family,run,expected in [('huge',a.huge,72),('validation',a.validation,15),('himvh',a.himvh,144)]:
    m=json.loads((run/'run_manifest.json').read_text());manifests[family]=m
    for name,h in m['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',m['git_commit']+':'+name])).hexdigest()==h==sha(ROOT/name),name
    ip=json.loads((run/'input_files_sha256.json').read_text());assert ip==dict(original=old,independent=meta['files'])
    complete=json.loads((run/'complete.json').read_text());assert complete['finished']==complete['expected']==expected and complete['failed']==0
    rr=json.loads((run/'records.json').read_text());assert len(rr)==expected;records+=rr
 # Score/report-only commits may differ; every frozen numerical source must be identical.
 assert all(m['source_sha256']==manifests['huge']['source_sha256'] for m in manifests.values())
 parent_old=json.loads((ROOT/'results/recent_repair_20261117/per_seed.json').read_text());seen=set();done_groups={};contexts={};probes={};verified=[];weightrows=[];eb=None
 for r in records:
    f,d,v,seed=r['family'],r['dataset'],r['variant'],r['seed'];key=(f,d,v,seed);assert key not in seen;seen.add(key)
    if f=='huge':
        group=a.huge/'graph'/d/'huge'/str(seed);folder=group/v
        if (d,seed) not in done_groups:
            phases={s:load_phase(a.inputs/'graphs'/d,s) for s in PHASES};role={s:v[0][0,:,0]>.5 for s,v in phases.items()};edges={s:bidirected_union(v[1],v[0].shape[1]) for s,v in phases.items()};xx={}
            for mode in ['legacy','dual','role']:xx[mode]={s:independent_roles(t[0],role[s],mode,phases['train'][0],role['train']) for s,t in phases.items()}
            with np.load(group/'components.npz',allow_pickle=False) as z:c,t=z['calibration'],z['test']
            teacher={s:raw_halo(xx['legacy'][s],edges[s]) for s in PHASES}
            with np.load(group/'train_teachers.npz',allow_pickle=False) as z:
                np.testing.assert_allclose(z['legacy'],teacher['train'],rtol=1e-6,atol=1e-6);np.testing.assert_allclose(z['dual'],raw_halo(xx['dual']['train'],edges['train']),rtol=1e-6,atol=1e-6)
            delta=[];parents=json.loads((group/'parents.json').read_text())
            for j,b in enumerate(['legacy','dual','decoupled','role_decoupled'],1):
                assert parents[b]==sha(group/(b+'.pt'));model=HUGE.load(group/(b+'.pt'),'cpu');assert model.steps==300 and model.seed==seed
                mode='legacy' if b=='legacy' else 'role' if b=='role_decoupled' else 'dual'
                for s,comp in [('calibration',c),('test',t)]:
                    prediction=model.score(xx[mode][s],edges[s])[role[s]];np.testing.assert_allclose(prediction,comp[:,j],rtol=5e-4,atol=2e-5);delta.append(float(np.max(np.abs(prediction-comp[:,j]))))
            for s,comp in [('calibration',c),('test',t)]:np.testing.assert_allclose(teacher[s][role[s]],comp[:,0],rtol=1e-6,atol=1e-6)
            with np.load(group/'calibrator.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['sorted'],np.sort(c,axis=0))
            values=[]
            for comp in [c,t]:
                z=tails(c,comp);values.append(dict(legacy_embed=comp[:,1],dual_embed=comp[:,2],decoupled_embed=comp[:,3],role_decoupled_embed=comp[:,4],teacher_only=comp[:,0],fusion=(z[:,0]+z[:,3])/2,native_fusion=(z[:,0]+z[:,2])/2,legacy_fusion=(z[:,0]+z[:,1])/2))
            with np.load(group/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
            if d!='cardsim':
                with np.load(a.old_AE/'graph'/d/'ae'/str(seed)/'indices.npz',allow_pickle=False) as z:
                    for k in z.files:np.testing.assert_array_equal(idx[k],z[k])
            for s in ['calibration','test']:np.testing.assert_array_equal(idx[s],np.flatnonzero(role[s]))
            done_groups[d,seed]=dict(values=values,parents=parents,delta=max(delta),y=np.load(a.inputs/'graphs'/d/'test.y.npy',allow_pickle=False)[role['test']])
        g=done_groups[d,seed];cal,score,gate=replay_metrics(r,folder,g['y']);np.testing.assert_array_equal(cal,g['values'][0][v]);np.testing.assert_array_equal(score,g['values'][1][v]);assert gate['parent_sha256']==r['parent_sha256'];assert r['parent_sha256']['checkpoints']==g['parents']
        for name,path in [('components','components.npz'),('calibrator','calibrator.npz'),('indices','indices.npz')]:assert r['parent_sha256'][name]==sha(group/path)
        info=dict(cpu_score_max_abs=g['delta'])
    elif f=='himvh':
        base,scorekind=v.split('_',1);parent=(a.validation if d=='cardsim' else a.old_parents)/'tabular'/d/'himvh'/base/str(seed);group=a.himvh/'tabular'/d/'himvh'/base/str(seed);folder=group/scorekind;ref=r['parent_sha256'];assert ref==json.loads((group/'parent_reference.json').read_text())
        for name,h in ref['files'].items():assert sha(parent/name)==h
        assert ref['calibrator']==sha(group/'calibrator.npz')
        if d!='cardsim':
            oldr=next(p for p in parent_old if (p['family'],p['dataset'],p['variant'],p['seed'])==('himvh',d,base,seed));assert oldr==json.loads((parent/'result.json').read_text());assert ref['files']['model.pt']==oldr['checkpoint_sha256']
        if (d,base,seed) not in probes:
            data=a.inputs/'tabular'/d;m=json.loads((data/'manifest.json').read_text());width=len(m['features']);rawx={s:np.memmap(data/f'{s}.x.f32',dtype='f4',mode='r',shape=(m['splits'][s]['rows'],width)) for s in PHASES}
            with np.load(parent/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
            ti=np.sort(np.random.default_rng(seed).choice(len(rawx['train']),min(8192,len(rawx['train'])),replace=False));ci=np.sort(np.random.default_rng(seed+100).choice(len(rawx['calibration']),min(100000,len(rawx['calibration'])),replace=False));np.testing.assert_array_equal(ti,idx['train']);np.testing.assert_array_equal(ci,idx['calibration']);np.testing.assert_array_equal(idx['test'],np.arange(len(rawx['test'])))
            pre=Preprocessor.load(parent/'preprocessor.npz');fit=Preprocessor().fit(rawx['train'][ti]);x={s:pre.transform(xx) for s,xx in rawx.items()}
            for s in PHASES:np.testing.assert_array_equal(x[s],fit.transform(rawx[s]))
            if d!='cardsim':
                oldpre=Preprocessor.load(a.old_AE/'tabular'/d/'ae'/str(seed)/'preprocessor.npz')
                for s in PHASES:np.testing.assert_array_equal(x[s],oldpre.transform(rawx[s]))
            with np.load(parent/'components.npz',allow_pickle=False) as z:ce,te=z['calibration'],z['test']
            gg=groups(d,width);center=np.median(x['train'][ti].astype('f8'),axis=0);be=np.array([((x['calibration'][ci][:,col].astype('f8')-center[col])**2).mean() for col in gg]);me=ce.mean(0);skill=np.minimum(1,np.maximum(0,1-me/np.maximum(be,1e-8)));dim=np.array([len(col) for col in gg],dtype=float)
            def norm(w):return w/w.sum() if w.sum()>0 else dim/dim.sum()
            weights=norm(dim*(.25+.75*skill));equal=norm(.25+.75*skill);hard=norm(dim*skill)
            with np.load(group/'calibrator.npz',allow_pickle=False) as z:
                for k,value in dict(center=center,null_error=be,model_error=me,skill=skill,widths=dim,weights=weights,equal=equal,hard=hard,sorted=np.sort(ce,axis=0)).items():close(z[k],value)
            vals=[]
            for comp in [ce,te]:
                z=tails(ce,comp);vals.append(dict(field=comp.mean(1),flat=np.average(comp,axis=1,weights=dim),tail=z.mean(1),tail_flat=np.average(z,axis=1,weights=dim),skill_tail=z@weights,skill_raw=comp@weights,skill_equal_tail=z@equal,skill_hard_tail=z@hard))
            if d!='cardsim':
                for name,oldv in [('field',base),('flat',base+'_flat')]:
                    with np.load(a.old_parents/'tabular'/d/'himvh'/oldv/str(seed)/'scores.npz',allow_pickle=False) as z:
                        for j,s in enumerate(['calibration','test']):np.testing.assert_array_equal(vals[j][name],z[s])
            model=(HIMGeneric if d=='cardsim' else HIMVHRepair).load(parent/'model.pt','cpu');va=SimpleNamespace(data=a.inputs/'tabular');vv=views(va,d,'test');rows=checkpoint_rows(len(te));pr=probe_him(model,x['test'],vv,rows,False,NativeContext(x['test'],vv));np.testing.assert_allclose(pr,te[rows],rtol=5e-4,atol=2e-5)
            for j in range(len(gg)):weightrows.append(dict(dataset=d,base=base,seed=seed,field=j,width=dim[j],null_MSE=be[j],model_MSE=me[j],skill=skill[j],weight=weights[j],width_prior_weight=dim[j]/dim.sum()))
            probes[d,base,seed]=dict(values=vals,y=np.fromfile(data/'test.y.i8',dtype='i1'),delta=float(np.max(np.abs(pr-te[rows]))))
            del x,rawx,model;gc.collect()
        g=probes[d,base,seed];cal,score,gate=replay_metrics(r,folder,g['y']);close(cal,g['values'][0][scorekind]);close(score,g['values'][1][scorekind]);assert gate['parent_sha256']==ref;info=dict(parent_component_CPU_probe_max_abs=g['delta'])
    else:
        method=v;track=r['track'];folder=a.validation/track/'cardsim'/'baseline'/method/str(seed);data=a.inputs/('graphs' if track=='graph' else 'tabular')/'cardsim';ref=r['parent_sha256']
        for n,h in ref.items():assert sha(folder/n)==h
        pre=Preprocessor.load(folder/'preprocessor.npz')
        if track=='tabular':
            m=json.loads((data/'manifest.json').read_text());width=len(m['features']);rawx={s:np.memmap(data/f'{s}.x.f32',dtype='f4',mode='r',shape=(m['splits'][s]['rows'],width)) for s in PHASES};ti=np.sort(np.random.default_rng(seed).choice(len(rawx['train']),min(8192,len(rawx['train'])),replace=False));ci=np.sort(np.random.default_rng(seed+100).choice(len(rawx['calibration']),min(100000,len(rawx['calibration'])),replace=False));fit=Preprocessor().fit(rawx['train'][ti]);x={s:pre.transform(xx) for s,xx in rawx.items()}
            for s in PHASES:np.testing.assert_array_equal(x[s],fit.transform(rawx[s]))
            model=NeuralDetector.load(folder/'model','cpu') if method=='vae' else ClassicalDetector.load(folder/'model');cs=model.score(x['calibration'][ci]);ts=model.score(x['test']);y=np.fromfile(data/'test.y.i8',dtype='i1')
        else:
            rawg={s:load_phase(data,s) for s in PHASES};role={s:v[0][0,:,0]>.5 for s,v in rawg.items()};fit=Preprocessor().fit(rawg['train'][0].reshape(-1,10));x={s:pre.transform(t[0].reshape(-1,10)).reshape(t[0].shape) for s,t in rawg.items()}
            for s in PHASES:np.testing.assert_array_equal(x[s],fit.transform(rawg[s][0].reshape(-1,10)).reshape(rawg[s][0].shape))
            if method=='audit_hcl_zero_adapted':
                model=AuditHCL.load(folder/'model.pt','cpu');cs=model.score(x['calibration'],rawg['calibration'][1],rawg['calibration'][2],role['calibration'])[role['calibration']];ts=model.score(x['test'],rawg['test'][1],rawg['test'][2],role['test'])[role['test']]
            else:
                if eb is None:
                    fetch_upstream(a.output/'scratch/upstream');eb={}
                    for s in ['calibration','test']:
                        flat=x[s].transpose(1,0,2).reshape(x[s].shape[1],-1);val,inf=score_graph(flat,rawg[s][1],a.output/'scratch'/s);eb[s]=(val[role[s]],inf)
                cs,ts=eb['calibration'][0],eb['test'][0]
                for s in ['calibration','test']:assert r['model_info'][s]['selected']==eb[s][1]['selected']
            y=np.load(data/'test.y.npy',allow_pickle=False)[role['test']]
        cal,score,gate=replay_metrics(r,folder,y);np.testing.assert_allclose(cs,cal,rtol=5e-4,atol=2e-5);np.testing.assert_allclose(ts,score,rtol=5e-4,atol=2e-5);assert gate['parent_sha256']==ref;info=dict(full_CPU_score_replay=True)
    verified.append(dict(family=f,dataset=d,variant=v,seed=seed,**info));print('VERIFIED',len(verified),'/231',f,d,v,seed,flush=True)
 expected={('huge',d,v,s) for d in TASKS for v in HUGE_VARIANTS for s in [42,43,44]}|{('himvh',d,b+'_'+v,s) for d in TASKS for b in ['m','cm'] for v in FIELD_VARIANTS for s in [42,43,44]}|{('baseline','cardsim',m,s) for m in ['hbos','lof','vae','ebgad_release_zero','audit_hcl_zero_adapted'] for s in [42,43,44]};assert seen==expected and len(records)==231
 a.output.mkdir(parents=True,exist_ok=True);write(a.output/'records.json',records);write(a.output/'run_manifests.json',manifests);write(a.output/'weights.json',weightrows)
 write(a.output/'verification.json',dict(status='passed',configurations=231,operating_records=462,new_fits=54,HUGE_new_fits=36,HIM_new_fits=6,baseline_new_fits=12,frozen_HIM_parents=18,original_input_files=101,independent_input_files=len(meta['files']),seconds=time.monotonic()-started,independent_source='CardSim synthetic world; frozen before generation; not real-bank external validation',frozen_numerical_sources_identical=True,verified=verified,checks=['all_frozen_sources','original101_exact_inputs','new26_exact_inputs','blind_simulation_provenance','disjoint_new_chronological_rows','reconstructed_native_graphs_and_account_labels','unchanged_original_indices_and_common_observations','independent_NumPy_HAL0','36_CPU_checkpoint_full_graph_replays','18_HIM_CPU_field_probes','independent_training_constant_baselines_and_skills','all_weights_and_CDF_scores','old_field_flat_bitwise_identity','baseline_full_CPU_replays_including_EB','all_score_parent_threshold_gates','all_POT_q99_and_metric_replays']))
 import shutil;shutil.rmtree(a.output/'scratch',ignore_errors=True);print('ALL VERIFIED231/231 and462 operating points',flush=True)
if __name__=='__main__':main()
