"""Actual shapes, no target parsing or financial outcome calculation."""
import argparse,gc,json,resource,shutil,time
from pathlib import Path
import numpy as np
import torch
from finad_geometry.experiment import inputs,source,huge_group,him_parent_group
from finad_geometry.models import TASKS
from finad_zero.graph_data import load_phase
from finad.preprocessing import Preprocessor
from finad_zero.audit_hcl import AuditHCL
from finad_zero.ebgad import fetch_upstream,score_graph
from finad_recent.data import write
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.device='cuda';a.output.mkdir(parents=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;inputs(a);started=time.monotonic();checks=[]
for task in TASKS:checks.append(huge_group(a,task,42,steps=2,qualify=True));gc.collect();torch.cuda.empty_cache()
for base in ['m','cm']:checks.append(him_parent_group(a,'cardsim',42,base,steps=2,qualify=True));gc.collect();torch.cuda.empty_cache()
x,e,active=load_phase(a.graphs/'cardsim','train');role=x[0,:,0]>.5;pre=Preprocessor().fit(x.reshape(-1,10));z=pre.transform(x.reshape(-1,10)).reshape(x.shape);m=AuditHCL(42,'cuda',2).fit(z,e,active,role);m.save(a.output/'audit.pt');o=AuditHCL.load(a.output/'audit.pt','cuda')
for k,v in m.model.state_dict().items():assert torch.equal(v,o.model.state_dict()[k])
assert np.isfinite(m.score(z,e,active,role)).all();checks.append(dict(kind='AuditHCL',train_nodes=x.shape[1]));del m,o;gc.collect();torch.cuda.empty_cache()
fetch_upstream(a.output/'scratch/upstream');flat=z.transpose(1,0,2).reshape(z.shape[1],-1);ss,info=score_graph(flat,e,a.output/'scratch/eb');assert np.isfinite(ss).all();checks.append(dict(kind='EBGAD',train_nodes=len(flat),selected=info['selected']));shutil.rmtree(a.output/'scratch')
write(a.output/'qualification.json',dict(status='passed',real_targets_read=False,financial_metrics=False,seconds=time.monotonic()-started,checks=checks,peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30,peak_rss_linux_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,source=source('smoke')))
print('ALL actual shape GPU and EB gates passed',flush=True)
