"""Independent full-data replay of the fixed equations, guards and metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc
from finad.preprocessing import Preprocessor
from finad.evaluation import metrics as subgroup_metrics

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--raw-runs',type=Path,required=True);p.add_argument('--original-runs',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
manifest=json.loads((a.runs/'run_manifest.json').read_text())
records=json.loads((a.runs/'records.json').read_text());assert len(records)==162
assert json.loads((a.runs/'complete.json').read_text())==dict(finished=162,failed=0,expected=162)
assert {(r['dataset'],r['seed'],r['variant']) for r in records}=={(d,s,v) for d in manifest['tasks'] for s in manifest['seeds'] for v in manifest['variants']}
for name,digest in manifest['source_sha256'].items():
    blob=subprocess.check_output(['git','show',manifest['git_commit']+':'+name],cwd=root)
    assert hashlib.sha256(blob).hexdigest()==digest
golden=json.loads((root/'results/zero_label_20261114/input_cache_files_sha256.json').read_text())
assert json.loads((a.runs/'input_files_sha256.json').read_text())==golden
for name,info in golden.items():assert (a.data/name).stat().st_size==info['bytes'] and sha(a.data/name)==info['sha256']
checks=[];strata=[];drivers=[];original=json.loads((a.original_runs/'records.json').read_text())
for d in manifest['tasks']:
    data=a.data/d;meta=json.loads((data/'manifest.json').read_text());width=len(meta['features'])
    x={s:np.memmap(data/f'{s}.x.f32',mode='r',dtype='float32',shape=(meta['splits'][s]['rows'],width)) for s in ['train','calibration','test']}
    for seed in manifest['seeds']:
        raw=a.raw_runs/d/str(seed);run=a.runs/d/str(seed)
        with np.load(raw/'indices.npz',allow_pickle=False) as z:ti,ci,ts=z['train'],z['calibration'],z['test']
        for split,index,offset,limit in [('train',ti,0,8192),('calibration',ci,100,100000)]:
            np.testing.assert_array_equal(index,np.sort(np.random.default_rng(seed+offset).choice(len(x[split]),min(limit,len(x[split])),replace=False)))
        np.testing.assert_array_equal(ts,np.arange(len(x['test'])))
        scaler=Preprocessor().fit(x['train'][ti]);stored=Preprocessor.load(raw/'preprocessor.npz')
        for name in ['keep','missing','center','scale']:np.testing.assert_array_equal(getattr(scaler,name),getattr(stored,name))
        values=np.concatenate([scaler.transform(x[s]) for s in ['train','calibration','test']]);nt,nc=len(x['train']),len(x['calibration'])
        train=values[ti].astype('float64');center=np.median(train,axis=0);q1,q3=np.quantile(train,[.25,.75],axis=0)
        b=(q3-q1)/1.349;stdtrain=train.std(axis=0);b=np.where(b>1e-8,b,np.where(stdtrain>1e-8,stdtrain,1.))
        prior=json.loads((run/'training_prior.json').read_text())
        np.testing.assert_array_equal(center,prior['center']);np.testing.assert_array_equal(b,prior['scale'])
        assert prior['floor_fraction']==.1 and prior['epsilon']==1e-6 and prior['lookback']==32
        starts=np.zeros(len(values),dtype=np.int64)
        if meta.get('temporal'):
            for split,offset in [('train',0),('calibration',nt),('test',nt+nc)]:
                seq=np.fromfile(data/f'{split}.seq.i32',dtype=np.int32);edges=np.r_[0,np.flatnonzero(seq[1:]!=seq[:-1])+1,len(seq)]
                for left,right in zip(edges[:-1],edges[1:]):starts[offset+left:offset+right]=offset+left
        components={}
        with np.load(raw/'forecasts.npz',allow_pickle=False) as pred:
            for phase,t in [('calibration',nt+ci),('test',nt+nc+ts)]:
                s=np.empty((len(t),width));count=np.minimum(32,t-starts[t]);last=values[np.maximum(t-1,0)].astype('float64');last[count==0]=center
                for begin in range(0,len(t),256):
                    pos=t[begin:begin+256];ix=pos[:,None]-np.arange(32,0,-1);h=values[np.maximum(ix,0)].astype('float64')
                    h[ix<starts[pos,None]]=0;s[begin:begin+len(pos)]=np.std(h,axis=1,ddof=0)
                components[phase]=dict(z=values[t].astype('float64'),prediction=pred[phase],s=s,count=count,last=last)
        y=np.fromfile(data/'test.y.i8',dtype=np.int8);known=y!=-1;yy=y[known]
        assert set(np.unique(y))<={-1,0,1}
        for variant in manifest['variants']:
            out=run/variant;r=next(r for r in records if (r['dataset'],r['seed'],r['variant'])==(d,seed,variant))
            for file,key in [('scores.npz','scores_sha256'),('feature_pot.json','feature_pot_sha256'),('thresholds.json','thresholds_sha256')]:assert sha(out/file)==r[key]
            assert sha(raw/'forecasts.npz')==r['frozen_forecasts_sha256'] and sha(raw/'preprocessor.npz')==r['preprocessor_sha256'] and sha(raw/'indices.npz')==r['indices_sha256']
            assert sha(run/'training_prior.json')==r['training_prior_sha256']
            fitted=json.loads((out/'feature_pot.json').read_text());active=np.asarray(fitted['active_indices'])
            assert fitted['quantile']==.90 and fitted['risk']==.01 and fitted['min_peaks']==30
            assert list(active)==sorted(set(active.tolist())) and len(active)>0
            assert fitted['active_features']==[meta['features'][i] for i in active]
            assert {x['index'] for x in fitted['failed']}|set(active)==set(range(width))
            scales=np.array([fitted['tails'][name]['threshold'] for name in fitted['active_features']])
            with np.load(out/'scores.npz',allow_pickle=False) as z:cal,test=z['calibration'],z['test']
            for phase,score in [('calibration',cal),('test',test)]:
                c=components[phase];pe=((c['z']-center)/(b+1e-6))**2
                if variant=='prior':e=pe
                else:
                    den=np.maximum(c['s'],.1*b)+1e-6 if variant in ('n','nc','persist') else c['s']+1e-6
                    base=c['last'] if variant=='persist' else c['prediction'];e=((c['z']-base)/den)**2
                    if variant in ('c','nc','persist'):
                        weight=c['count'][:,None]/32;e=weight*e+(1-weight)*pe
                ratio=e[:,active]/scales;np.testing.assert_allclose(ratio.max(1),score,rtol=1e-12,atol=1e-12)
                if phase=='calibration':
                    for j,name in zip(active,fitted['active_features']):
                        tail=fitted['tails'][name];u=float(np.quantile(e[:,j],.90));peaks=e[e[:,j]>u,j]-u
                        assert tail['n_peaks']==len(peaks) and tail['n_calibration']==len(e) and len(peaks)>=30 and np.ptp(peaks)>0
                        assert np.isfinite(genpareto.logpdf(peaks,tail['shape'],loc=0,scale=tail['scale'])).all()
                        threshold=u+genpareto.isf(.01*len(e)/len(peaks),tail['shape'],loc=0,scale=tail['scale'])
                        np.testing.assert_allclose(threshold,tail['threshold'],rtol=1e-12,atol=1e-12)
                else:
                    arg=active[ratio.argmax(1)];constant=c['s'][np.arange(len(arg)),arg]<=1e-8
            th=json.loads((out/'thresholds.json').read_text());assert th['quantile']['threshold']==np.quantile(cal,.99)
            pot=th['pot'];accept=False;reasons=[]
            if pot['status']=='ok':
                u=np.quantile(cal,.85);peaks=cal[cal>u]-u
                assert len(peaks)>=20 and np.ptp(peaks)>0
                expected=u+genpareto.isf(.01*len(cal)/len(peaks),pot['shape'],loc=0,scale=pot['scale'])
                np.testing.assert_allclose(expected,pot['threshold'],rtol=1e-12,atol=1e-12)
                rate=float(np.mean(cal>=pot['threshold']))
                if pot['shape']>=1:reasons.append('shape_not_below_one')
                if not .005<=rate<=.02:reasons.append('calibration_alarm_rate_outside_fixed_interval')
                accept=not reasons
            else:reasons=['original_fit_failed']
            guarded=th['guarded_pot'];assert guarded['rejection_reasons']==reasons
            assert guarded['selected']==('original_pot' if accept else 'quantile_fallback')
            assert guarded['threshold']==(pot['threshold'] if accept else th['quantile']['threshold'])
            assert guarded['calibration_alarm_fraction']==float(np.mean(cal>=guarded['threshold']))
            m=r['metrics'];pcurve,rcurve,_=precision_recall_curve(yy,test[known])
            for name,value in [('average_precision',average_precision_score(yy,test[known])),('roc_auc',roc_auc_score(yy,test[known])),('pr_auc_trapezoid',auc(rcurve,pcurve))]:np.testing.assert_allclose(m[name],value,rtol=1e-12,atol=1e-12)
            assert m['evaluated']==len(yy) and m['unknown_excluded']==int((~known).sum()) and m['positives']==int(yy.sum()) and m['normal']==int((yy==0).sum())
            cold=components['test']['count']<32
            for op,t in th.items():
                v=m['operating_points'][op]
                if t['status']!='ok':assert v is None;continue
                alarm=test[known]>=t['threshold'];tp=int(((yy==1)&alarm).sum());fp=int(((yy==0)&alarm).sum());fn=int(((yy==1)&~alarm).sum());tn=int(((yy==0)&~alarm).sum())
                assert [v[k] for k in ['tp','fp','fn','tn']]==[tp,fp,fn,tn]
                vals=dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),fpr=fp/(fp+tn),alarm_fraction=float(alarm.mean()))
                for k,value in vals.items():np.testing.assert_allclose(v[k],value,rtol=1e-12,atol=1e-12)
                full_alarm=test>=t['threshold'];normal_fp=(y==0)&full_alarm
                drivers.append(dict(dataset=d,seed=seed,variant=variant,operating_point=op,fp=fp,tp=tp,
                                    cold_fp=int((normal_fp&cold).sum()),warm_fp=int((normal_fp&~cold).sum()),
                                    near_constant_driver_fp=int((normal_fp&constant).sum()),
                                    cold_tp=int(((y==1)&full_alarm&cold).sum()),warm_tp=int(((y==1)&full_alarm&~cold).sum())))
            for group,mask in [('all',np.ones(len(y),dtype=bool)),('cold_start',cold),('warm',~cold)]:
                if not mask.any():continue
                sub=subgroup_metrics(y[mask],test[mask],th)
                for op,values_op in sub['operating_points'].items():
                    row=dict(dataset=d,seed=seed,variant=variant,stratum=group,operating_point=op,scored=int(mask.sum()),**{k:v for k,v in sub.items() if k!='operating_points'})
                    for k in ['precision','recall','f1','fpr','tp','fp','fn','tn']:row[k]=values_op[k] if values_op else None
                    if row['positives']==0:row['recall']=None;row['f1']=None
                    strata.append(row)
            if variant=='legacy':
                old=next(r for r in original if (r['dataset'],r['seed'])==(d,seed))['metrics']
                for key in old:
                    if key=='operating_points':
                        for op,value in old[key].items():
                            if value is None:assert m[key][op] is None
                            else:
                                for k,v in value.items():np.testing.assert_allclose(m[key][op][k],v,rtol=1e-12,atol=1e-12)
                    else:np.testing.assert_allclose(m[key],old[key],rtol=1e-12,atol=1e-12)
            checks.append(dict(dataset=d,seed=seed,variant=variant,test_rows=len(y),matched_original=variant=='legacy'))
        print('VERIFIED_SIX_FIXED_VARIANTS',d,seed,f'{len(checks)}/162',flush=True)
def savecsv(name,rows):
    with (a.runs/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
savecsv('cold_warm_metrics.csv',strata);savecsv('score_driver_diagnostics.csv',drivers)
write(a.runs/'verification.json',dict(status='passed',runs=162,operating_point_records=486,exact_input_files=81,
      original_27_reproduced=True,all_scores_independently_replayed=True,all_guards_calibration_only_verified=True,
      unchanged_full_test_coverage=True,new_backbone_inference=False,verified=checks))
print('STABLE_FULL_COHORT_INDEPENDENTLY_VERIFIED',flush=True)
