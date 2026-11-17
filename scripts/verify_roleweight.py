"""Independent replay of features, frozen fields, thresholds and metrics."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc
from finad_recent.data import ROOT,sha,write,verify_inputs,PHASES,GRAPH,transaction_keys
from finad.preprocessing import Preprocessor
from finad_recent.huge import HUGE,bidirected_union
from finad_zero.graph_data import load_phase
from finad_repair.himvh import HIMVHRepair
from finad_repair.hypergraph import NativeContext
from verify_repair import checkpoint_rows,probe_him,close


def replay_metrics(r,folder,y):
    with np.load(folder/'scores.npz',allow_pickle=False) as z:cal=z['calibration'];score=z['test']
    assert np.isfinite(cal).all() and np.isfinite(score).all()
    m=r['metrics'];known=y!=-1;t=y[known];s=score[known]
    for k,v in dict(evaluated=len(t),unknown_excluded=int((~known).sum()),positives=int(t.sum()),normal=int((t==0).sum()),prevalence=float(t.mean())).items():close(m[k],v)
    p,rec,_=precision_recall_curve(t,s)
    for k,v in dict(average_precision=average_precision_score(t,s),pr_auc_trapezoid=auc(rec,p),roc_auc=roc_auc_score(t,s)).items():close(m[k],v)
    thresholds=json.loads((folder/'thresholds.json').read_text())
    u=np.quantile(cal,.85);peaks=cal[cal>u]-u
    for k,v in thresholds.items():
        if v['status']!='ok':
            assert k=='pot' and (len(peaks)<20 or np.ptp(peaks)==0) and m['operating_points'][k] is None
            continue
        if k=='quantile':expected=np.quantile(cal,.99)
        else:
            assert len(peaks)>=20 and np.ptp(peaks)>0
            shape,_,scale=genpareto.fit(peaks,floc=0);close(v['shape'],shape);close(v['scale'],scale)
            close(v['initial_threshold'],u);assert v['risk']==.01 and v['initial_quantile']==.85
            expected=u+genpareto.isf(.01*len(cal)/len(peaks),shape,loc=0,scale=scale)
        close(v['threshold'],expected)
        a=s>=expected;tp=int(np.sum(a&(t==1)));fp=int(np.sum(a&(t==0)));fn=int(np.sum(~a&(t==1)));tn=int(np.sum(~a&(t==0)))
        vals=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),fpr=fp/(fp+tn),alarm_fraction=float(a.mean()))
        for f,value in vals.items():close(m['operating_points'][k][f],value)
    gate=json.loads((folder/'label_access_gate.json').read_text())
    assert gate['status']=='scores_and_thresholds_saved_before_labels'
    assert gate['scores_sha256']==sha(folder/'scores.npz')==r['score_sha256']
    assert gate['thresholds_sha256']==sha(folder/'thresholds.json')
    return cal,score,gate


def independent_fields(cal,test,widths):
    # Independent right-continuous empirical CDF via unique values/counts,
    # not FieldScorer's sorted-array search implementation.
    n=len(cal);z=[]
    for x in [cal,test]:
        cols=[]
        for j in range(cal.shape[1]):
            values,counts=np.unique(cal[:,j],return_counts=True);below=np.r_[0,np.cumsum(counts)]
            less=below[np.searchsorted(values,x[:,j],side='left')]
            cols.append(-np.log((1+n-less)/(n+1)))
        z.append(np.column_stack(cols))
    rawmean=cal.mean(0);mean=np.where(rawmean>1e-8,rawmean,1.);med=np.median(cal,0);med=np.where(med>1e-8,med,mean)
    # np.corrcoef is a second implementation of covariance normalization.
    with np.errstate(invalid='ignore',divide='ignore'):c=np.corrcoef(z[0],rowvar=False)
    c=np.nan_to_num(c);np.fill_diagonal(c,0);weights=1/(1+np.clip(c,0,1).sum(1));weights/=weights.sum()
    result=[]
    for x,v in zip([cal,test],z):result.append(dict(field=x.mean(1),flat=np.average(x,axis=1,weights=widths),mean_scale=(x/mean).mean(1),median_scale=(x/med).mean(1),tail=v.mean(1),tail_flat=np.average(v,axis=1,weights=widths),tail_max=v.max(1),tail_corr=v@weights))
    return result,dict(sorted=np.sort(cal,axis=0),widths=widths,mean=mean,median=med,corr_weights=weights)


def independent_roles(raw,role,variant,train,trainrole):
    # Refit every statistic from train graph; align with explicit indices.
    common=Preprocessor().fit(train.reshape(-1,10));original=common.transform(raw.reshape(-1,10)).reshape(raw.shape)
    flat=lambda x:x.transpose(1,0,2).reshape(raw.shape[1],-1)
    numeric=flat(original[:,:,2:])
    if variant=='legacy':return flat(original)
    if variant=='numeric':return numeric
    def directional(v,r):return np.stack([v[s,np.arange(v.shape[1])[:,None],np.where(r[:,None],np.array([2,4,6,8]),np.array([3,5,7,9]))] for s in range(4)])
    c=directional(raw,role);tc=directional(train,trainrole)
    pool=Preprocessor().fit(tc.reshape(-1,4));align=flat(pool.transform(c.reshape(-1,4)).reshape(4,-1,4))
    rolez=np.zeros_like(c);center=np.zeros_like(c)
    for r in [False,True]:
        pre=Preprocessor().fit(tc[:,trainrole==r].reshape(-1,4))
        rolez[:,role==r]=pre.transform(c[:,role==r].reshape(-1,4)).reshape(4,-1,4)
        center[:,role==r]=np.clip((c[:,role==r].astype('f8')-pre.center)/pool.scale,-8,8)
    rolez=flat(rolez);center=flat(center)
    if variant in ['aligned','centered','role']:return dict(aligned=align,centered=center,role=rolez)[variant]
    return np.column_stack([numeric/np.sqrt(32),dict(dual=align,dual_center=center,dual_role=rolez)[variant]/4]).astype('f4')


def main():
    p=argparse.ArgumentParser();p.add_argument('--huge',type=Path,required=True);p.add_argument('--himvh',type=Path,required=True);p.add_argument('--parents',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic();torch.set_num_threads(4)
    inputs=verify_inputs(a.inputs/'tabular',a.inputs/'graphs');assert len(inputs)==101
    manifests={};records=[];verified=[]
    frozen=json.loads((ROOT/'results/recent_repair_20261117/per_seed.json').read_text())
    assert json.loads((ROOT/'results/recent_repair_20261117/verification.json').read_text())['status']=='passed'
    for family,run,expected in [('huge',a.huge,48),('himvh',a.himvh,96)]:
        m=json.loads((run/'run_manifest.json').read_text());manifests[family]=m
        assert json.loads((run/'input_files_sha256.json').read_text())==inputs
        assert json.loads((run/'complete.json').read_text())==dict(finished=expected,expected=expected,failed=0)
        for f,h in m['source_sha256'].items():
            assert hashlib.sha256(subprocess.check_output(['git','show',m['git_commit']+':'+f])).hexdigest()==h==sha(ROOT/f)
        values=json.loads((run/'records.json').read_text());assert len(values)==expected;records+=values
    assert manifests['huge']['git_commit']==manifests['himvh']['git_commit']
    contexts={};probes={};seen=set()
    for r in records:
        family,task,v,seed=r['family'],r['dataset'],r['variant'],r['seed'];key=(family,task,v,seed);assert key not in seen;seen.add(key)
        if family=='huge':
            folder=a.huge/'graph'/task/'huge'/v/str(seed)
            if task not in contexts:
                phases={s:load_phase(a.inputs/'graphs'/task,s) for s in PHASES};roles={s:x[0][0,:,0]>.5 for s,x in phases.items()};edges={s:bidirected_union(x[1],x[0].shape[1]) for s,x in phases.items()};contexts[task]=(phases,roles,edges)
            phases,roles,edges=contexts[task];y=np.load(a.inputs/'graphs'/task/'test.y.npy',allow_pickle=False)[roles['test']]
            cal,score,gate=replay_metrics(r,folder,y);assert gate['checkpoint_sha256']==sha(folder/'model.pt')==r['checkpoint_sha256']
            model=HUGE.load(folder/'model.pt','cpu');assert model.steps==300
            with np.load(folder/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
            with np.load(a.old/'graph'/task/'ae'/str(seed)/'indices.npz',allow_pickle=False) as z:
                for k in z.files:np.testing.assert_array_equal(idx[k],z[k])
            for s in ['calibration','test']:np.testing.assert_array_equal(idx[s],np.flatnonzero(roles[s]))
            delta=[]
            for s in PHASES:
                x=independent_roles(phases[s][0],roles[s],v,phases['train'][0],roles['train'])
                assert hashlib.sha256(x.tobytes()).hexdigest()==r['checks']['feature_sha256'][s]
                if s!='train':
                    value=model.score(x,edges[s])[roles[s]];saved=cal if s=='calibration' else score
                    np.testing.assert_allclose(value,saved,rtol=5e-4,atol=2e-5);delta.append(float(np.max(np.abs(value-saved))))
            assert sha(a.inputs/'graphs'/task/'test.y.npy')==r['checks']['test_target_sha256']
            verified.append(dict(family=family,dataset=task,variant=v,seed=seed,cpu_score_max_abs=max(delta)))
        else:
            base,scoring=v.split('_',1);parent=a.parents/'tabular'/task/'himvh'/base/str(seed);group=a.himvh/'tabular'/task/'himvh'/base/str(seed);folder=group/scoring
            parent_record=next(p for p in frozen if (p['family'],p['dataset'],p['variant'],p['seed'])==('himvh',task,base,seed))
            assert json.loads((parent/'result.json').read_text())==parent_record
            ref=r['parent_reference'];assert ref==json.loads((group/'parent_reference.json').read_text())
            for f,h in ref['files_sha256'].items():assert sha(parent/f)==h
            assert ref['parent_source_commit']=='09de7f313d57470293f1309ed611914a88cbe62c' and ref['fit_reused']
            key2=(task,base,seed)
            if key2 not in probes:
                with np.load(parent/'components.npz',allow_pickle=False) as z:c,t=z['calibration'],z['test']
                widths=np.array([1,8,4,32,16] if task=='banksim' else [1,16,16,16],dtype=float)
                scores,parameters=independent_fields(c,t,widths)
                with np.load(group/'calibrator.npz',allow_pickle=False) as z:
                    for k,value in parameters.items():close(z[k],value)
                # Source input and preprocessor remain bitwise paired with AE.
                data=a.inputs/'tabular'/task;meta=json.loads((data/'manifest.json').read_text());width=len(meta['features']);pre=Preprocessor.load(parent/'preprocessor.npz');oldpre=Preprocessor.load(a.old/'tabular'/task/'ae'/str(seed)/'preprocessor.npz')
                with np.load(parent/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
                with np.load(a.old/'tabular'/task/'ae'/str(seed)/'indices.npz',allow_pickle=False) as z:
                    for k in z.files:np.testing.assert_array_equal(idx[k],z[k])
                for s in PHASES:
                    raw=np.memmap(data/f'{s}.x.f32',dtype='f4',mode='r',shape=(meta['splits'][s]['rows'],width));np.testing.assert_array_equal(pre.transform(raw),oldpre.transform(raw))
                np.testing.assert_array_equal(idx['test'],np.arange(len(t)))
                raw=np.memmap(data/'test.x.f32',dtype='f4',mode='r',shape=(len(t),width));x=pre.transform(raw)
                if task not in probes:probes[task]=transaction_keys(task,'test')[0]
                views=probes[task];model=HIMVHRepair.load(parent/'model.pt','cpu');rows=checkpoint_rows(len(t));comp=probe_him(model,x,views,rows,False,NativeContext(x,views));np.testing.assert_allclose(comp,t[rows],rtol=5e-4,atol=2e-5)
                for name,control in [('field',base),('flat',base+'_flat')]:
                    with np.load(a.parents/'tabular'/task/'himvh'/control/str(seed)/'scores.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(scores[0][name],z['calibration']);np.testing.assert_array_equal(scores[1][name],z['test'])
                probes[key2]=(scores,idx,float(np.max(np.abs(comp-t[rows]))))
            scores,idx,error=probes[key2];y=np.fromfile(a.inputs/'tabular'/task/'test.y.i8',dtype=np.int8);cal,score,gate=replay_metrics(r,folder,y)
            close(cal,scores[0][scoring]);close(score,scores[1][scoring]);assert len(cal)==len(idx['calibration'])
            assert gate['checkpoint_sha256']==ref['files_sha256']['model.pt']==r['checkpoint_sha256'];assert gate['components_sha256']==ref['files_sha256']['components.npz'];assert gate['calibrator_sha256']==sha(group/'calibrator.npz')==ref['calibrator_sha256']
            assert sha(a.inputs/'tabular'/task/'test.y.i8')==r['checks']['test_target_sha256']
            verified.append(dict(family=family,dataset=task,variant=v,seed=seed,parent_component_cpu_probe_max_abs=error))
        print('VERIFIED',len(verified),'/144',family,task,v,seed,flush=True)
    from finad_roleweight.adapters import VARIANTS as R
    from finad_roleweight.fields import VARIANTS as F
    assert seen=={('huge',t,v,s) for t in GRAPH for v in R for s in [42,43,44]}|{('himvh',t,b+'_'+v,s) for t in GRAPH for b in ['m','cm'] for v in F for s in [42,43,44]}
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'records.json',records);write(a.output/'run_manifests.json',manifests)
    write(a.output/'verification.json',dict(status='passed',runs=144,operating_records=288,actual_new_fits=48,frozen_parent_checkpoints=12,parent_component_reuse=True,input_files=101,seconds=time.monotonic()-started,verified=verified,checks=['frozen_source','101_exact_input_hashes','same_AE_indices_and_cohorts','shared_float32_observations','independent_role_features_train_only','full_graph_CPU_score_replay','12_frozen_HIM_component_CPU_probes','independent_empirical_tail_counts_all_rows','all_weights_and_normalizers','old_field_flat_bitwise_identity','parent_model_component_calibrator_score_threshold_hash_gates','independent_POT_q99_refit_and_full_metrics']))
    print('ALL VERIFIED144 configurations /288 operating points',flush=True)
if __name__=='__main__':main()
