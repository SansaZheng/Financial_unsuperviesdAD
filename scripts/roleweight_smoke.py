"""Actual train graph shapes, all adapter widths, no labels or metrics."""
import argparse,copy,resource,time
from pathlib import Path
import numpy as np
import torch
from finad_recent.data import GRAPH,write,verify_inputs
from finad_recent.huge import HUGE,bidirected_union
from finad_zero.graph_data import load_phase
from finad_roleweight.adapters import RoleAdapter,VARIANTS
from finad_roleweight.experiment import manifest
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
verify_inputs(a.data,a.graphs);results=[]
for task in GRAPH:
 raw,e,_=load_phase(a.graphs/task,'train');role=raw[0,:,0]>.5;edges=bidirected_union(e,raw.shape[1])
 for variant in VARIANTS:
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();x=RoleAdapter(variant).fit(raw,role).transform(raw,role)
    model=HUGE(x.shape[1],42,'cuda',2).fit(x,edges);path=a.output/f'{task}_{variant}.pt';model.save(path);loaded=HUGE.load(path,'cpu')
    for k,v in model.net.state_dict().items():assert torch.equal(v.cpu(),loaded.net.state_dict()[k]),k
    cpu=copy.deepcopy(model);cpu.net=cpu.net.cpu();cpu.device='cpu';np.testing.assert_array_equal(cpu.score(x,edges),loaded.score(x,edges))
    assert np.isfinite(model.score(x,edges)).all()
    results.append(dict(task=task,variant=variant,seconds=time.monotonic()-started,peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30,peak_rss_linux_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    del model,loaded,cpu;torch.cuda.empty_cache();print('QUALIFIED',task,variant,flush=True)
write(a.output/'qualification.json',dict(status='passed',financial_metrics=False,real_targets_read=False,train_features_only=True,models=results,source=manifest('huge')))
