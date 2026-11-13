"""Recompute metrics from saved scores/targets using independent explicit counts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,auc

p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--data',type=Path,required=True);p.add_argument('--graph-data',type=Path,required=True)
args=p.parse_args();records=json.loads((args.results/'records.json').read_text())
seen=set();verified=[]
for r in records:
    key=(r['track'],r['dataset'],r['method'],r['seed']);assert key not in seen;seen.add(key)
    root=args.results/r['track']/r['dataset']/r['method']/str(r['seed'])
    assert hashlib.sha256((root/'scores.npz').read_bytes()).hexdigest()==r['score_sha256']
    with np.load(root/'scores.npz',allow_pickle=False) as z: cal,score=z['calibration'],z['test']
    if r['track']=='tabular':y=np.fromfile(args.data/r['dataset']/'test.y.i8',dtype=np.int8)
    else:
        with np.load(args.graph_data/r['dataset']/'test.graph.npz',allow_pickle=False) as z: role=z['x'][0,:,0]>.5
        y=np.load(args.graph_data/r['dataset']/'test.y.npy',allow_pickle=False)[role]
    assert y.shape==score.shape and np.isfinite(score).all() and np.isfinite(cal).all()
    known=y!=-1;y=y[known];s=score[known];m=r['metrics']
    assert m['unknown_excluded']==int((~known).sum()) and m['positives']==int(y.sum()) and m['normal']==int((y==0).sum())
    if len(np.unique(y))==2:
        pr,re,_=precision_recall_curve(y,s)
        for name,value in [('average_precision',average_precision_score(y,s)),('roc_auc',roc_auc_score(y,s)),('pr_auc_trapezoid',auc(re,pr))]:
            assert np.isclose(m[name],value,rtol=1e-12,atol=1e-12),(key,name)
    thresholds=json.loads((root/'thresholds.json').read_text())
    for name,th in thresholds.items():
        if th['status']!='ok':assert m['operating_points'][name] is None;continue
        if name=='quantile':assert th['threshold']==float(np.quantile(cal,.99))
        else:
            u=float(np.quantile(cal,.85));peaks=cal[cal>u]-u
            assert len(peaks)>=20 and np.ptp(peaks)>0
            value=u+genpareto.isf(.01*len(cal)/len(peaks),th['shape'],loc=0,scale=th['scale'])
            assert np.isclose(value,th['threshold'],rtol=1e-12,atol=1e-12)
        alarm=s>=th['threshold'];tp=int(((y==1)&alarm).sum());fp=int(((y==0)&alarm).sum())
        fn=int(((y==1)&~alarm).sum());tn=int(((y==0)&~alarm).sum());op=m['operating_points'][name]
        assert [op[k] for k in ('tp','fp','fn','tn')]==[tp,fp,fn,tn],key
        for k,value in [('precision',tp/(tp+fp) if tp+fp else 0.),('recall',tp/(tp+fn) if tp+fn else 0.),
                        ('f1',2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.),('fpr',fp/(fp+tn) if fp+tn else None)]:
            assert op[k] is None if value is None else np.isclose(op[k],value,rtol=1e-12,atol=1e-12),(key,k)
    verified.append('/'.join(map(str,key)))
complete=json.loads((args.results/'complete.json').read_text())
assert complete==dict(finished=144,failed=0,expected=144),complete
(args.results/'verification.json').write_text(json.dumps({'status':'passed','runs':len(verified),'checks':
 ['score_SHA256','target_alignment','unknown_mask','AP','trapezoid_PR_AUC','ROC_AUC','quantile','POT_formula','confusion_counts','precision_recall_F1_FPR'],
 'verified':verified},indent=2)+'\n')
print('ZERO_LABEL_144_RUNS_VERIFIED',flush=True)
