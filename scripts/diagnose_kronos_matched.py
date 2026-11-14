"""Post-hoc descriptive diagnostics; never refit scores, models or thresholds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from finad.preprocessing import Preprocessor
from finad.evaluation import metrics

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
records=json.loads((a.runs/'records.json').read_text());strata=[];drivers=[];overview=[]
for r in records:
    data=a.data/r['dataset'];run=a.runs/r['dataset']/str(r['seed'])
    meta=json.loads((data/'manifest.json').read_text());width=len(meta['features'])
    pre=Preprocessor.load(run/'preprocessor.npz')
    x={s:np.memmap(data/f'{s}.x.f32',mode='r',dtype='float32',shape=(meta['splits'][s]['rows'],width)) for s in ('train','calibration','test')}
    values=np.concatenate([pre.transform(x[s]) for s in ('train','calibration','test')]);nt,nc=len(x['train']),len(x['calibration'])
    starts=np.zeros(len(values),dtype=np.int64)
    if meta.get('temporal'):
        for split,offset in [('train',0),('calibration',nt),('test',nt+nc)]:
            seq=np.fromfile(data/f'{split}.seq.i32',dtype=np.int32);boundaries=np.r_[0,np.flatnonzero(np.diff(seq))+1,len(seq)]
            for left,right in zip(boundaries[:-1],boundaries[1:]):starts[offset+left:offset+right]=offset+left
    positions=nt+nc+np.arange(len(x['test']));cold=positions-starts[positions]<32
    with np.load(run/'scores.npz',allow_pickle=False) as z:score=z['test']
    with np.load(run/'forecasts.npz',allow_pickle=False) as z:predicted=z['test']
    fit=json.loads((run/'feature_pot.json').read_text()) if (run/'feature_pot.json').exists() else None
    if fit:
        active=np.array(fit['active_indices']);scales=np.array([fit['tails'][f]['threshold'] for f in fit['active_features']])
    dominant=np.empty(len(score),dtype=np.int32);near_constant=np.empty(len(score),dtype=bool)
    for begin in range(0,len(score),256):
        t=positions[begin:begin+256];ix=t[:,None]-np.arange(32,0,-1);valid=ix>=starts[t,None]
        h=values[np.maximum(ix,0)].astype('float64');h[~valid]=0;std=h.std(axis=1)
        error=((values[t].astype('float64')-predicted[begin:begin+len(t)])/(std+1e-6))**2
        if fit:
            ratio=error[:,active]/scales;arg=active[ratio.argmax(1)];expected=ratio.max(1)
        else:arg=error.argmax(1);expected=error.max(1)
        np.testing.assert_allclose(expected,score[begin:begin+len(t)],rtol=1e-12,atol=1e-12)
        dominant[begin:begin+len(t)]=arg;near_constant[begin:begin+len(t)]=std[np.arange(len(t)),arg]<=1e-8
    y=np.fromfile(data/'test.y.i8',dtype=np.int8);thresholds=json.loads((run/'thresholds.json').read_text())
    for label,mask in [('all',np.ones(len(score),dtype=bool)),('cold_start',cold),('warm',~cold)]:
        if not mask.any():continue
        result=metrics(y[mask],score[mask],thresholds)
        for op,values_op in result['operating_points'].items():
            row=dict(dataset=r['dataset'],seed=r['seed'],stratum=label,operating_point=op,threshold_unchanged=True,scored=int(mask.sum()))
            row.update({k:v for k,v in result.items() if k!='operating_points'})
            for key in ['precision','recall','f1','fpr','tp','fp','fn','tn']:row[key]=values_op[key] if values_op else None
            # A zero-positive subgroup does not establish anomaly recall or F1 ability.
            if row['positives']==0:row['recall']=None;row['f1']=None
            strata.append(row)
    known=y!=-1
    info=dict(dataset=r['dataset'],seed=r['seed'],test_rows=len(score),cold_start_rows=int(cold.sum()),dominant_near_constant_history_rows=int(near_constant.sum()),dominant_near_constant_history_fraction=float(near_constant.mean()))
    for op,th in thresholds.items():
        alarm=score>=th['threshold'] if th['status']=='ok' else np.zeros(len(score),dtype=bool)
        fp=known&(y==0)&alarm
        info[op+'_known_false_alarms']=int(fp.sum()) if th['status']=='ok' else None
        info[op+'_false_alarms_near_constant_history_fraction']=float(near_constant[fp].mean()) if fp.any() else None
    overview.append(info)
    for j,name in enumerate(meta['features']):
        mask=dominant==j
        drivers.append(dict(dataset=r['dataset'],seed=r['seed'],feature=name,dominant_rows=int(mask.sum()),dominant_share=float(mask.mean()),known_normal_dominant_rows=int((mask&(y==0)).sum()),known_anomaly_dominant_rows=int((mask&(y==1)).sum())))
    print('DESCRIPTIVE_DIAGNOSTICS',r['dataset'],r['seed'],flush=True)
def writecsv(name,rows):
    with (a.output/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
writecsv('cold_warm_metrics.csv',strata);writecsv('dominant_features.csv',drivers);writecsv('score_driver_diagnostics.csv',overview)
(a.output/'diagnostics_protocol.json').write_text(json.dumps(dict(type='Post-hoc descriptive strata and score drivers, not a causal ablation',no_refitting=True,no_model_selection=True,thresholds_unchanged=True,cold_start_rule='fewer than 32 past rows in the same stream/case/event',near_constant_rule='dominant residual feature has past-window population std <= 1e-8 in shared feature units',argmax_ties='First feature in the fixed schema order',main_evaluation='Full test remains primary; warm-only metrics are never substituted',runs=len(records)),indent=2)+'\n')
