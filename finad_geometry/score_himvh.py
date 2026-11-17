"""Freeze old/new parent models, fit only unlabeled field reliability weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from finad_recent.data import PHASES,sha,write
from finad.preprocessing import Preprocessor
from .experiment import inputs,source,finish,record
from .models import SkillScorer,groups,FIELD_VARIANTS,TASKS

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--old-parents',type=Path,required=True);p.add_argument('--new-parents',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 write(a.output/'input_files_sha256.json',inputs(a));write(a.output/'run_manifest.json',source('himvh'));records=[]
 for task in TASKS:
  for seed in [42,43,44]:
   for base in ['m','cm']:
    start=time.monotonic();parent=(a.new_parents if task=='cardsim' else a.old_parents)/'tabular'/task/'himvh'/base/str(seed);out=a.output/'tabular'/task/'himvh'/base/str(seed);out.mkdir(parents=True)
    with np.load(parent/'indices.npz',allow_pickle=False) as z:idx={k:z[k] for k in z.files}
    pre=Preprocessor.load(parent/'preprocessor.npz');data=a.data/task;meta=json.loads((data/'manifest.json').read_text());width=len(meta['features']);raw={s:np.memmap(data/f'{s}.x.f32',mode='r',dtype='f4',shape=(meta['splits'][s]['rows'],width)) for s in ['train','calibration']}
    tx=pre.transform(raw['train'][idx['train']]);cx=pre.transform(raw['calibration'][idx['calibration']])
    with np.load(parent/'components.npz',allow_pickle=False) as z:components={s:z[s] for s in ['calibration','test']}
    scorer=SkillScorer().fit(tx,cx,components['calibration'],groups(task,width));scorer.save(out/'calibrator.npz');scores={s:scorer.all_scores(c) for s,c in components.items()}
    if task!='cardsim':
     for name,control in [('field',base),('flat',base+'_flat')]:
      with np.load(a.old_parents/'tabular'/task/'himvh'/control/str(seed)/'scores.npz',allow_pickle=False) as z:
       for s in components:np.testing.assert_array_equal(scores[s][name],z[s])
    parent_info=dict(parent_dataset=task,parent_base=base,parent_seed=seed,parent_frozen=True,parent_source='same_new_source' if task=='cardsim' else '09de7f313d57470293f1309ed611914a88cbe62c',files={n:sha(parent/n) for n in ['model.pt','components.npz','indices.npz','preprocessor.npz']},calibrator=sha(out/'calibrator.npz'))
    write(out/'parent_reference.json',parent_info)
    for v in FIELD_VARIANTS:
     folder=out/v;folder.mkdir();m=finish(folder,scores['calibration'][v],scores['test'][v],parent_info,lambda:np.fromfile(data/'test.y.i8',dtype='i1'));records.append(record(folder,'tabular',task,'himvh',base+'_'+v,seed,m,dict(fit_reused=True,scoring_only=True,skill=scorer.skill.tolist(),weights=scorer.weights.tolist(),widths=scorer.widths.tolist()),start,parent_info))
    print('DONE',len(records),'/144',task,base,seed,'seconds',round(time.monotonic()-start,2),flush=True)
 write(a.output/'records.json',records);write(a.output/'complete.json',dict(finished=144,expected=144,failed=0))
if __name__=='__main__':main()
