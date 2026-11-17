"""Fixed role factorial / frozen-parent field-score factorial execution."""
import argparse,copy,gc,json,platform,time
from pathlib import Path
import numpy as np
import scipy,sklearn,torch
from threadpoolctl import threadpool_limits
from finad_recent.data import ROOT,PHASES,GRAPH,sha,write,verify_inputs
from finad_recent.huge import HUGE,bidirected_union
from finad_recent.experiment import finish
from finad_zero.graph_data import load_phase
from finad_zero.experiment import code_commit
from finad.thresholds import calibrate
from finad.evaluation import metrics
from .adapters import RoleAdapter,VARIANTS as ROLES
from .fields import FieldScorer,VARIANTS as FIELDS

PARENT_SOURCE='09de7f313d57470293f1309ed611914a88cbe62c'

def manifest(family):
    files=[f for package in ['finad_roleweight','finad_recent','finad_repair'] for f in (ROOT/package).glob('*.py')]+[ROOT/'finad'/n for n in ['preprocessing.py','thresholds.py','evaluation.py']]
    return dict(git_commit=code_commit(),source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)},family=family,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,seeds=[42,43,44],primary='dual' if family=='huge' else 'cm_tail')

def graph_run(graphs,task,variant,seed,out,device,steps=300):
    data=graphs/task;phases={s:load_phase(data,s) for s in PHASES};roles={s:v[0][0,:,0]>.5 for s,v in phases.items()}
    adapter=RoleAdapter(variant).fit(phases['train'][0],roles['train']);adapter.save(out)
    x={s:adapter.transform(v[0],roles[s]) for s,v in phases.items()};restored=RoleAdapter.load(out)
    for s in PHASES:np.testing.assert_array_equal(x[s],restored.transform(phases[s][0],roles[s]))
    edges={s:bidirected_union(v[1],v[0].shape[1]) for s,v in phases.items()}
    eligible=np.flatnonzero(roles['train']);ti=np.sort(np.random.default_rng(seed).choice(eligible,min(8192,len(eligible)),replace=False))
    np.savez(out/'indices.npz',train=ti,calibration=np.flatnonzero(roles['calibration']),test=np.flatnonzero(roles['test']))
    model=HUGE(x['train'].shape[1],seed,device,steps).fit(x['train'],edges['train']);model.save(out/'model.pt')
    loaded=HUGE.load(out/'model.pt',device)
    for k,v in model.net.state_dict().items():assert torch.equal(v.cpu(),loaded.net.state_dict()[k].cpu()),k
    # Strict weight identity + deterministic CPU reload; CUDA scatter isn't bitwise.
    m=copy.deepcopy(model);m.net=m.net.cpu();m.device='cpu'
    l=HUGE.load(out/'model.pt','cpu')
    np.testing.assert_array_equal(m.score(x['train'],edges['train']),l.score(x['train'],edges['train']))
    scores={s:model.score(x[s],edges[s])[roles[s]] for s in ['calibration','test']}
    result=finish(out,scores['calibration'],scores['test'],lambda:np.load(data/'test.y.npy',allow_pickle=False)[roles['test']])
    return result,dict(**model.info,adapter=variant,fit_reused=False,input_width=x['train'].shape[1]),dict(test_target_sha256=sha(data/'test.y.npy'),checkpoint_cpu_exact=True,feature_sha256={s:__import__('hashlib').sha256(x[s].tobytes()).hexdigest() for s in PHASES})

def run_himvh(a):
    records=[]
    for task in GRAPH:
      for seed in [42,43,44]:
       for base in ['m','cm']:
        started=time.monotonic();parent=a.parents/'tabular'/task/'himvh'/base/str(seed);group=a.output/'tabular'/task/'himvh'/base/str(seed);group.mkdir(parents=True,exist_ok=False)
        old=json.loads((parent/'result.json').read_text())
        assert old['checkpoint_sha256']==sha(parent/'model.pt')
        assert old['score_sha256']==sha(parent/'scores.npz')
        with np.load(parent/'components.npz',allow_pickle=False) as z:cal=z['calibration'];test=z['test']
        widths=[1,8,4,32,16] if task=='banksim' else [1,16,16,16]
        scorer=FieldScorer().fit(cal,widths);scorer.save(group/'calibrator.npz');again=FieldScorer.load(group/'calibrator.npz')
        cs,ts=scorer.all_scores(cal),scorer.all_scores(test)
        for k,v in again.all_scores(test).items():np.testing.assert_array_equal(ts[k],v)
        for name,control in [('field',base),('flat',base+'_flat')]:
            with np.load(a.parents/'tabular'/task/'himvh'/control/str(seed)/'scores.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(cs[name],z['calibration']);np.testing.assert_array_equal(ts[name],z['test'])
        refs={name:sha(parent/name) for name in ['model.pt','components.npz','indices.npz','preprocessor.npz','result.json','label_access_gate.json']}
        reference=dict(parent_source_commit=PARENT_SOURCE,parent_relative=str(parent.relative_to(a.parents)),files_sha256=refs,calibrator_sha256=sha(group/'calibrator.npz'),fit_reused=True,components_reused=True)
        write(group/'parent_reference.json',reference)
        for variant in FIELDS:
            out=group/variant;out.mkdir()
            np.savez_compressed(out/'scores.npz',calibration=cs[variant],test=ts[variant]);thresholds=calibrate(cs[variant]);write(out/'thresholds.json',thresholds)
            write(out/'label_access_gate.json',dict(checkpoint_sha256=refs['model.pt'],components_sha256=refs['components.npz'],calibrator_sha256=reference['calibrator_sha256'],scores_sha256=sha(out/'scores.npz'),thresholds_sha256=sha(out/'thresholds.json'),status='scores_and_thresholds_saved_before_labels'))
            y=np.fromfile(a.data/task/'test.y.i8',dtype=np.int8)
            record=dict(track='tabular',dataset=task,family='himvh',variant=base+'_'+variant,method='himvh_roleweight_'+base+'_'+variant,seed=seed,seconds=time.monotonic()-started,metrics=metrics(y,ts[variant],thresholds),checkpoint_sha256=refs['model.pt'],score_sha256=sha(out/'scores.npz'),parent_reference=reference,model_info=dict(fit_reused=True,scoring_only=True,base=base,scoring=variant),checks=dict(test_target_sha256=sha(a.data/task/'test.y.i8')))
            write(out/'result.json',record);records.append(record)
        print(f'DONE {len(records)}/96 himvh/{task}/{base}/{seed} group_seconds={time.monotonic()-started:.1f}',flush=True)
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=['huge','himvh'],required=True);p.add_argument('--parents',type=Path);p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'run_manifest.json').exists():raise ValueError('Fresh output required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    write(a.output/'input_files_sha256.json',verify_inputs(a.data,a.graphs));write(a.output/'run_manifest.json',manifest(a.family))
    # Immutable baseline implementation, not just pinned inputs.
    old=json.loads((ROOT/'results/recent_zero_20261117/run_manifest.json').read_text())
    for name,h in old['source_sha256'].items():
        if name.startswith(('finad_recent/','finad/')):assert sha(ROOT/name)==h,name
    with threadpool_limits(limits=4):
      if a.family=='himvh':records=run_himvh(a)
      else:
        records=[]
        for task in GRAPH:
         for seed in [42,43,44]:
          for variant in ROLES:
            out=a.output/'graph'/task/'huge'/variant/str(seed);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
            if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            result,info,checks=graph_run(a.graphs,task,variant,seed,out,a.device)
            record=dict(track='graph',dataset=task,family='huge',variant=variant,method='huge_roleweight_'+variant,seed=seed,seconds=time.monotonic()-started,metrics=result,model_info=info,checks=checks,checkpoint_sha256=sha(out/'model.pt'),score_sha256=sha(out/'scores.npz'),peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else None)
            write(out/'result.json',record);records.append(record);write(a.output/'progress.json',dict(finished=len(records),expected=48,failed=0));print(f'DONE {len(records)}/48 huge/{task}/{variant}/{seed} seconds={record["seconds"]:.1f}',flush=True)
            gc.collect()
            if a.device.startswith('cuda'):torch.cuda.empty_cache()
    write(a.output/'records.json',records);write(a.output/'complete.json',dict(finished=len(records),expected=48 if a.family=='huge' else 96,failed=0))
if __name__=='__main__':main()
