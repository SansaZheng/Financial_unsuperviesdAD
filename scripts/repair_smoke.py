"""Train-feature-only real-shape GPU qualification, no target parsing."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from finad.preprocessing import Preprocessor
from finad_recent.data import write, transaction_keys, verify_inputs
from finad_recent.huge import HUGE,bidirected_union
from finad_recent.sdfmm import sequence_starts
from finad_zero.graph_data import load_phase
from finad_repair.roles import RoleNormalizer
from finad_repair.himvh import HIMVHRepair
from finad_repair.sdfmm import SDRepair
from finad_repair.experiment import reload_net


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    verify_inputs(a.data,a.graphs);results={}
    for family in ['huge','himvh','sdfmm']:
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        if family=='huge':
            x,e,_=load_phase(a.graphs/'banksim','train');role=x[0,:,0]>.5
            z=RoleNormalizer().fit(x,role).transform(x,role);edges=bidirected_union(e,x.shape[1])
            m=HUGE(16,device=a.device,steps=2).fit(z,edges);m.save(a.output/'huge.pt');other=HUGE.load(a.output/'huge.pt',a.device)
            first=m.score(z,edges);again=other.score(z,edges)
        else:
            task='banksim' if family=='himvh' else 'sd_stock';data=a.data/task;meta=json.loads((data/'manifest.json').read_text());n=meta['splits']['train']['rows'];p=len(meta['features'])
            raw=np.memmap(data/'train.x.f32',mode='r',dtype='float32',shape=(n,p));ti=np.sort(np.random.default_rng(42).choice(n,min(8192,n),replace=False));pre=Preprocessor().fit(raw[ti])
            if family=='himvh':
                x=pre.transform(raw);views,_=transaction_keys(task,'train')
                m=HIMVHRepair(p,task,device=a.device,native=True,consistent=True,steps=2).fit(x,views,ti)
                m.save(a.output/'himvh.pt');other=HIMVHRepair.load(a.output/'himvh.pt',a.device)
                error=reload_net(m,other,x,a.device)
                view,_=transaction_keys(task,'train',np.arange(120000))
                score=m.components(x[:120000],view)
                assert score.shape==(120000,5) and np.isfinite(score).all()
                first=np.array([error]);again=np.zeros(1)
            else:
                starts=sequence_starts(np.fromfile(data/'train.seq.i32',dtype=np.int32))
                m=SDRepair(p,task,device=a.device,endpoint=True,generator=True,steps=2).fit(raw,starts,ti,pre)
                m.save(a.output/'sdfmm.pt');other=SDRepair.load(a.output/'sdfmm.pt',a.device)
                first=m.score_raw(raw,starts,ti[:64],pre);again=other.score_raw(raw,starts,ti[:64],pre)
        np.testing.assert_allclose(first,again,rtol=1e-5,atol=1e-6)
        results[family]=dict(seconds=time.monotonic()-started,checkpoint_reload_max_abs=float(np.max(np.abs(first-again))),
                             peak_cuda_gb=torch.cuda.max_memory_allocated()/2**30,peak_rss_linux_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        del m,other
        import gc;gc.collect();torch.cuda.empty_cache();print('SMOKE '+family+' passed',flush=True)
    write(a.output/'qualification.json',dict(status='passed',financial_metrics=False,real_targets_read=False,train_features_only=True,models=results,torch=torch.__version__))


if __name__=='__main__':main()
