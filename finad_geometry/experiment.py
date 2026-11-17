"""All fixed variants; durable score gates precede target access."""
import argparse,copy,gc,hashlib,json,platform,shutil,time
from pathlib import Path
import numpy as np
import scipy,sklearn,torch
from threadpoolctl import threadpool_limits
from finad_recent.data import ROOT,PHASES,sha,write,verify_inputs
from finad_zero.experiment import code_commit,score_chunks
from finad_zero.graph_data import load_phase
from finad_recent.huge import HUGE,halo,bidirected_union
from finad_roleweight.adapters import RoleAdapter
from finad_recent.experiment import load_tabular
from finad_recent.data import transaction_keys
from finad_repair.experiment import reload_net
from finad.classical import ClassicalDetector
from finad.neural import NeuralDetector
from finad.preprocessing import Preprocessor
from finad.thresholds import calibrate
from finad.evaluation import metrics
from finad_zero.audit_hcl import AuditHCL
from finad_zero.ebgad import fetch_upstream,score_graph
from .models import GeometryHUGE,HIMGeneric,SkillScorer,TailMap,groups,TASKS,HUGE_VARIANTS,FIELD_VARIANTS

PARENTS=('legacy','dual','decoupled','role_decoupled')

def inputs(a):
    old=verify_inputs(a.data,a.graphs);m=json.loads((ROOT/'docs/geometry_cardsim_archive.json').read_text())
    for name,h in m['files'].items():
        p=(a.data.parent/name);assert p.stat().st_size==h['bytes'] and sha(p)==h['sha256'],name
    return dict(original=old,independent=m['files'])

def source(family):
    paths=[p for pkg in ['finad_geometry','finad_roleweight','finad_recent','finad_repair','finad_zero','finad'] for p in (ROOT/pkg).glob('*.py')]+[ROOT/'scripts/prepare_repo_data.py',ROOT/'docs/geometry_protocol_CN.md',ROOT/'docs/geometry_cardsim_archive.json']
    return dict(git_commit=code_commit(),family=family,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,seeds=[42,43,44],primary=dict(huge='fusion',himvh='cm_skill_tail'),independent_outcomes_inspected=False)

def views(a,task,phase,indices=None):
    if task!='cardsim':return transaction_keys(task,phase,indices)[0]
    with np.load(a.data/task/f'{phase}.views.npz',allow_pickle=False) as z:v=[z[f'e{i}'] for i in range(4)]
    if indices is None:return v
    # Recreate the original sampled training windows from view-group membership,
    # not an induced subset of full-graph edges.
    from finad_recent.himvh import hyperedges
    # Saved keys allow exact regrouping; input views alone lose singleton keys.
    with np.load(a.data/task/f'{phase}.keys.npz',allow_pickle=False) as z:return [hyperedges(z[f'k{i}'][indices],4) for i in range(4)]

def finish(out,cs,ts,parent,read_labels):
    if not np.isfinite(cs).all() or not np.isfinite(ts).all():raise FloatingPointError('Nonfinite scores')
    np.savez_compressed(out/'scores.npz',calibration=cs,test=ts);thresholds=calibrate(cs);write(out/'thresholds.json',thresholds)
    write(out/'label_access_gate.json',dict(parent_sha256=parent,scores_sha256=sha(out/'scores.npz'),thresholds_sha256=sha(out/'thresholds.json'),status='scores_and_thresholds_saved_before_labels'))
    return metrics(read_labels(),ts,thresholds)

def record(out,track,task,family,variant,seed,m,info,started,parent):
    r=dict(track=track,dataset=task,family=family,variant=variant,method=f'{family}_geometry_{variant}',seed=seed,metrics=m,model_info=info,seconds=time.monotonic()-started,parent_sha256=parent,score_sha256=sha(out/'scores.npz'))
    write(out/'result.json',r);return r

def huge_group(a,task,seed,steps=300,qualify=False):
    data=a.graphs/task;raw={s:load_phase(data,s) for s in PHASES};roles={s:v[0][0,:,0]>.5 for s,v in raw.items()};edges={s:bidirected_union(v[1],v[0].shape[1]) for s,v in raw.items()}
    root=a.output/'graph'/task/'huge'/str(seed);root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    eligible=np.flatnonzero(roles['train']);ti=np.sort(np.random.default_rng(seed).choice(eligible,min(8192,len(eligible)),replace=False));np.savez(root/'indices.npz',train=ti,calibration=np.flatnonzero(roles['calibration']),test=np.flatnonzero(roles['test']))
    features={}
    for mode in ['legacy','dual','role']:
        pre=RoleAdapter(mode).fit(raw['train'][0],roles['train']);pre.save(root/mode)
        features[mode]={s:pre.transform(v[0],roles[s]) for s,v in raw.items()}
    teachers={}
    for mode in ['legacy','dual']:
        teachers[mode]={s:halo(torch.as_tensor(features[mode][s]),torch.as_tensor(edges[s])).numpy() for s in PHASES}
    np.savez_compressed(root/'train_teachers.npz',legacy=teachers['legacy']['train'],dual=teachers['dual']['train'])
    components={s:[teachers['legacy'][s][roles[s]]] for s in ['calibration','test']};parents={};infos={}
    for base in PARENTS:
        mode='role' if base=='role_decoupled' else 'dual' if base in ['dual','decoupled'] else 'legacy';teacher='dual' if base=='dual' else 'legacy';x=features[mode]
        model=GeometryHUGE(x['train'].shape[1],seed,a.device,steps).fit_teacher(x['train'],edges['train'],teachers[teacher]['train']);path=root/(base+'.pt');model.save(path);parents[base]=sha(path)
        restored=HUGE.load(path,'cpu');cpu=copy.deepcopy(model);cpu.net=cpu.net.cpu();cpu.device='cpu'
        for k,v in model.net.state_dict().items():assert torch.equal(v.cpu(),restored.net.state_dict()[k]),k
        np.testing.assert_array_equal(cpu.score(x['train'],edges['train']),restored.score(x['train'],edges['train']))
        for s in components:components[s].append(model.score(x[s],edges[s])[roles[s]])
        infos[base]=dict(**model.info,student=mode,teacher=teacher,input_width=model.width)
        del model,cpu,restored;gc.collect()
        if a.device.startswith('cuda'):torch.cuda.empty_cache()
    components={s:np.column_stack(v) for s,v in components.items()};np.savez_compressed(root/'components.npz',**components);tail=TailMap().fit(components['calibration']);np.savez_compressed(root/'calibrator.npz',sorted=tail.sorted)
    all_scores={}
    for s,c in components.items():
        z=tail.transform(c);all_scores[s]=dict(legacy_embed=c[:,1],dual_embed=c[:,2],decoupled_embed=c[:,3],role_decoupled_embed=c[:,4],teacher_only=c[:,0],fusion=(z[:,0]+z[:,3])/2,native_fusion=(z[:,0]+z[:,2])/2,legacy_fusion=(z[:,0]+z[:,1])/2)
    write(root/'model_info.json',infos);write(root/'parents.json',parents)
    if qualify:return dict(task=task,nodes={s:v[0].shape[1] for s,v in raw.items()},parents=parents,seconds=time.monotonic()-started,score_shapes={s:list(v.shape) for s,v in components.items()})
    records=[]
    for variant in HUGE_VARIANTS:
        out=root/variant;out.mkdir();parent=dict(checkpoints=parents,components=sha(root/'components.npz'),calibrator=sha(root/'calibrator.npz'),indices=sha(root/'indices.npz'))
        m=finish(out,all_scores['calibration'][variant],all_scores['test'][variant],parent,lambda:np.load(data/'test.y.npy',allow_pickle=False)[roles['test']])
        records.append(record(out,'graph',task,'huge',variant,seed,m,dict(steps=steps,fit_reused=variant not in [b+'_embed' for b in PARENTS],parents=infos,teacher_only=variant=='teacher_only'),started,parent))
    return records

def him_parent_group(a,task,seed,base,steps=800,qualify=False):
    root=a.output/'tabular'/task/'himvh'/base/str(seed);root.mkdir(parents=True,exist_ok=False);meta,x,ti,ci=load_tabular(a.data/task,root,seed);native={s:views(a,task,s) for s in PHASES};c=base=='cm'
    model=HIMGeneric(x['train'].shape[1],task,seed,a.device,native=c,steps=steps);trainx=x['train'] if c else x['train'][ti];selected=native['train'] if c else views(a,task,'train',ti)
    model.fit(trainx,selected,ti if c else np.arange(len(ti)),None if c else ti);model.save(root/'model.pt');loaded=HIMGeneric.load(root/'model.pt',a.device);reload_net(model,loaded,x['train'][ti],a.device)
    if qualify:
        with np.load(a.data/task/'calibration.views.npz',allow_pickle=False) as z:vv=[z[f'e{i}'] for i in range(4)]
        comp=model.components(x['calibration'],vv);assert np.isfinite(comp).all();return dict(task=task,base=base,rows=len(x['train']),groups=len(model.groups),calibration_rows=len(comp))
    comp={s:model.components(x[s],native[s])[ci] if s=='calibration' else model.components(x[s],native[s]) for s in ['calibration','test']};np.savez_compressed(root/'components.npz',**comp);write(root/'parent_info.json',dict(checkpoint_sha256=sha(root/'model.pt'),components_sha256=sha(root/'components.npz'),info=model.info));return root

def baseline(a,method,seed,cache):
    started=time.monotonic();graph=method in ['ebgad_release_zero','audit_hcl_zero_adapted'];track='graph' if graph else 'tabular';root=a.output/track/'cardsim'/'baseline'/method/str(seed);root.mkdir(parents=True,exist_ok=False);data=(a.graphs if graph else a.data)/'cardsim'
    if graph:
        phases={s:load_phase(data,s) for s in PHASES};role={s:v[0][0,:,0]>.5 for s,v in phases.items()};pre=Preprocessor().fit(phases['train'][0].reshape(-1,10));pre.save(root/'preprocessor.npz');z={s:pre.transform(v[0].reshape(-1,10)).reshape(v[0].shape) for s,v in phases.items()};ei=np.flatnonzero(role['train']);ti=np.sort(np.random.default_rng(seed).choice(ei,min(8192,len(ei)),replace=False));np.savez(root/'indices.npz',train=ti,calibration=np.flatnonzero(role['calibration']),test=np.flatnonzero(role['test']))
        if method=='audit_hcl_zero_adapted':
            model=AuditHCL(seed,a.device,200).fit(z['train'],phases['train'][1],phases['train'][2],role['train']);model.save(root/'model.pt');loaded=AuditHCL.load(root/'model.pt',a.device)
            for k,v in model.model.state_dict().items():assert torch.equal(v,loaded.model.state_dict()[k])
            scores={s:model.score(z[s],phases[s][1],phases[s][2],role[s])[role[s]] for s in ['calibration','test']};info=model.info;parents={'model.pt':sha(root/'model.pt')}
        else:
            if 'eb' not in cache:
                fetch_upstream(a.output/'scratch/upstream');cache['eb']={}
                for s in ['calibration','test']:
                    flat=z[s].transpose(1,0,2).reshape(z[s].shape[1],-1);ss,im=score_graph(flat,phases[s][1],a.output/'scratch'/s);cache['eb'][s]=(ss[role[s]],im)
            scores={s:cache['eb'][s][0] for s in ['calibration','test']};info={s:cache['eb'][s][1] for s in scores};write(root/'model_spec.json',info);parents={'model_spec.json':sha(root/'model_spec.json')}
        read=lambda:np.load(data/'test.y.npy',allow_pickle=False)[role['test']]
    else:
        meta,x,ti,ci=load_tabular(data,root,seed);model=NeuralDetector('vae',seed,a.device,steps=400) if method=='vae' else ClassicalDetector(method,seed);model.fit(x['train'][ti]);model.save(root/'model');loaded=type(model).load(root/'model',device=a.device) if method=='vae' else type(model).load(root/'model')
        np.testing.assert_allclose(model.score(x['test'][:64]),loaded.score(x['test'][:64]),rtol=1e-5,atol=1e-6)
        scores=dict(calibration=score_chunks(model,x['calibration'][ci]),test=score_chunks(model,x['test']));info=model.info;parents={str(p.relative_to(root)):sha(p) for p in root.glob('model*') if p.is_file()};read=lambda:np.fromfile(data/'test.y.i8',dtype='i1')
    m=finish(root,scores['calibration'],scores['test'],parents,read);return record(root,track,'cardsim','baseline',method,seed,m,info,started,parents)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--family',choices=['huge','validation'],required=True);p.add_argument('--device',default='cuda');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'run_manifest.json').exists():raise ValueError('Fresh directory required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    write(a.output/'input_files_sha256.json',inputs(a));write(a.output/'run_manifest.json',source(a.family));records=[];parents=[]
    with threadpool_limits(limits=4):
      if a.family=='huge':
        for task in TASKS:
         for seed in [42,43,44]:
            records+=huge_group(a,task,seed);write(a.output/'progress.json',dict(finished=len(records),expected=72,failed=0));print('DONE HUGE',len(records),'/72',task,seed,flush=True)
      else:
        cache={}
        for seed in [42,43,44]:
         for base in ['m','cm']:
            root=him_parent_group(a,'cardsim',seed,base);parents.append(str(root.relative_to(a.output)));print('DONE HIM parent',base,seed,flush=True)
         for method in ['hbos','lof','vae','ebgad_release_zero','audit_hcl_zero_adapted']:
            records.append(baseline(a,method,seed,cache));print('DONE baseline',len(records),'/15',method,seed,flush=True)
         gc.collect()
         if a.device.startswith('cuda'):torch.cuda.empty_cache()
        shutil.rmtree(a.output/'scratch',ignore_errors=True);write(a.output/'HIM_parents.json',parents)
    write(a.output/'records.json',records);write(a.output/'complete.json',dict(finished=len(records),expected=72 if a.family=='huge' else 15,failed=0,HIM_parents=len(parents)))
if __name__=='__main__':main()
