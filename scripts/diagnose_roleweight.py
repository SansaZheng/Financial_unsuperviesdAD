"""Posthoc interpretation only: never changes scoring, weights or primaries."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from finad_recent.data import GRAPH,PHASES,sha,write
from finad_zero.graph_data import load_phase
from finad_recent.huge import halo,bidirected_union
from finad_roleweight.adapters import RoleAdapter,VARIANTS
from finad_roleweight.fields import FieldScorer
from report_recent_zero import savecsv
p=argparse.ArgumentParser();p.add_argument('--huge',type=Path,required=True);p.add_argument('--himvh',type=Path,required=True);p.add_argument('--parents',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert json.loads((a.output/'verification.json').read_text())['status']=='passed';torch.set_num_threads(4)
fields=dict(banksim=['amount','age','gender','category','merchant'],s_ffsd=['amount','location','type','target']);rows=[];graph=[]
for task in GRAPH:
 y=np.fromfile(a.inputs/'tabular'/task/'test.y.i8',dtype='i1');known=y!=-1
 for base in ['m','cm']:
  for seed in [42,43,44]:
   parent=a.parents/'tabular'/task/'himvh'/base/str(seed);group=a.himvh/'tabular'/task/'himvh'/base/str(seed);scorer=FieldScorer.load(group/'calibrator.npz')
   with np.load(parent/'components.npz',allow_pickle=False) as z:c,t=z['calibration'],z['test']
   cz,tz=scorer.evidence(c),scorer.evidence(t);cm,tm=c.mean(0),t.mean(0);ct,tt=cz.mean(0),tz.mean(0)
   for j,name in enumerate(fields[task]):
    upper=t[:,j]>scorer.sorted[-1,j]
    rows.append(dict(dataset=task,base=base,seed=seed,field=name,raw_cal_share=cm[j]/cm.sum(),tail_cal_share=ct[j]/ct.sum(),raw_test_share=tm[j]/tm.sum(),tail_test_share=tt[j]/tt.sum(),corr_weight=scorer.corr_weights[j],raw_field_AP=average_precision_score(y[known],t[known,j]),tail_field_AP=average_precision_score(y[known],tz[known,j]),test_upper_saturation_fraction=float(upper.mean()),test_tail_mean_normal=float(tz[y==0,j].mean()),test_tail_mean_fraud=float(tz[y==1,j].mean())))
 raw={s:load_phase(a.inputs/'graphs'/task,s) for s in PHASES};role={s:v[0][0,:,0]>.5 for s,v in raw.items()};y=np.load(a.inputs/'graphs'/task/'test.y.npy',allow_pickle=False)[role['test']];known=y!=-1
 for variant in VARIANTS:
  adapter=RoleAdapter.load(a.huge/'graph'/task/'huge'/variant/'42')
  x=adapter.transform(raw['test'][0],role['test']);e=bidirected_union(raw['test'][1],len(x));value=halo(torch.as_tensor(x),torch.as_tensor(e)).numpy()[role['test']]
  graph.append(dict(dataset=task,variant=variant,diagnostic='test_input_HALO_not_a_trained_detector_or_candidate',teacher_geometry_AP=float(average_precision_score(y[known],value[known])),teacher_geometry_ROC=float(roc_auc_score(y[known],value[known])),payer_mean_norm=float(np.linalg.norm(x[role['test']],axis=1).mean()),payee_mean_norm=float(np.linalg.norm(x[~role['test']],axis=1).mean())))
savecsv(a.output/'posthoc_field_diagnostics.csv',rows);savecsv(a.output/'posthoc_role_geometry.csv',graph)
write(a.output/'posthoc_scope.json',dict(purpose='evaluation_only_interpretation',after_all_runs_and_verification=True,model_selection=False,score_or_threshold_changes=False,primaries_unchanged=['huge_dual','himvh_cm_tail'],labels_used='test labels only for posthoc field AP and input teacher geometry diagnostics; not for scoring',field_mapping='native semantic feature groups; raw error and tail error, not raw feature utility',source_files={f:sha(Path(f)) for f in ['finad_roleweight/adapters.py','finad_roleweight/fields.py','scripts/diagnose_roleweight.py']}))
print('Posthoc only:54 field diagnostic rows,16 role geometry rows')
