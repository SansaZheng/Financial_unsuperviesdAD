"""Frozen geometry teacher, baseline-compatible losses and unlabeled skill."""
import numpy as np
import torch
from finad_recent.huge import HUGE,halo,loop_edges,similarity,rank_loss,alignment
from finad_recent.himvh import HimvhNet
from finad_repair.himvh import HIMVHRepair
from finad_repair.hypergraph import feature_groups
from finad_roleweight.fields import FieldScorer

HUGE_VARIANTS=('legacy_embed','dual_embed','decoupled_embed','role_decoupled_embed','teacher_only','fusion','native_fusion','legacy_fusion')
FIELD_VARIANTS=('field','flat','tail','tail_flat','skill_tail','skill_raw','skill_equal_tail','skill_hard_tail')
TASKS=('banksim','s_ffsd','cardsim')

def groups(task,width):
    if task!='cardsim':return feature_groups(task,width)
    assert width==24
    return [np.arange(a,b) for a,b in [(0,1),(1,2),(2,4),(4,6),(6,22),(22,24)]]

class GeometryHUGE(HUGE):
    def fit_teacher(self,x,edges,teacher):
        x=torch.as_tensor(x,device=self.device);e=torch.as_tensor(edges,device=self.device,dtype=torch.long)
        h=torch.as_tensor(teacher,device=self.device,dtype=x.dtype)
        if h.shape!=(len(x),) or not torch.isfinite(h).all():raise ValueError('Invalid fixed teacher')
        e=loop_edges(e,len(x));opt=torch.optim.Adam(self.net.parameters(),lr=.0005);gen=torch.Generator(device=self.device).manual_seed(self.seed);losses=[];self.net.train()
        for _ in range(self.steps):
            opt.zero_grad();mlp,gnn=self.net(x,e);sample=torch.randperm(len(x),generator=gen,device=self.device)[:512];loss=x.new_zeros(())
            for z in (mlp,gnn):
                pos,neg=similarity(z,e,True);pred=-pos;loss=loss+(rank_loss(pred[sample],h[sample])+torch.nn.functional.softplus(pred+neg).mean())/2
            loss=loss+.5*alignment(mlp,gnn,e)
            if not torch.isfinite(loss):raise FloatingPointError('HUGE loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
        self.info.update(loss_first=losses[0],loss_last=losses[-1],teacher_external=True)
        return self

class HIMGeneric(HIMVHRepair):
    def __init__(self,width,task,seed=42,device='cpu',native=False,consistent=True,steps=800):
        self.width,self.task,self.seed,self.device=width,task,seed,device;self.native,self.consistent,self.steps=native,consistent,steps;self.groups=groups(task,width)
        torch.manual_seed(seed);self.net=HimvhNet(width,4).to(device)
        self.info=dict(native_context=native,consistent_masked_field=consistent,steps=steps,batch=128,target_exposures=steps*128,hidden=256,layers=3,window=4,lr=.001,decoder_normalization='original BatchNorm',supervised_labels=False)

class TailMap:
    def fit(self,c):
        c=np.asarray(c,dtype='f8')
        if c.ndim!=2 or not len(c) or not np.isfinite(c).all():raise ValueError('Need finite calibration matrix')
        self.sorted=np.sort(c,axis=0);return self
    def transform(self,x):
        x=np.asarray(x,dtype='f8');n=len(self.sorted)
        if x.ndim!=2 or x.shape[1]!=self.sorted.shape[1] or not np.isfinite(x).all():raise ValueError('Score matrix mismatch')
        return np.column_stack([-np.log((1+n-np.searchsorted(self.sorted[:,g],x[:,g],side='left'))/(n+1)) for g in range(x.shape[1])])

class SkillScorer:
    def fit(self,train,cal,errors,field_groups):
        self.groups=field_groups;self.center=np.median(np.asarray(train,dtype='f8'),axis=0)
        baseline=(np.asarray(cal,dtype='f8')-self.center)**2
        self.null_error=np.array([baseline[:,g].mean() for g in self.groups]);self.model_error=np.asarray(errors).mean(0)
        if len(errors)!=len(cal) or not np.isfinite(self.model_error).all() or (self.model_error<0).any():raise ValueError('Invalid errors')
        self.widths=np.array([len(g) for g in self.groups],dtype='f8');self.skill=np.clip(1-self.model_error/np.maximum(self.null_error,1e-8),0,1)
        def norm(w):return w/w.sum() if w.sum()>0 else self.widths/self.widths.sum()
        self.weights=norm(self.widths*(.25+.75*self.skill));self.equal=norm(.25+.75*self.skill);self.hard=norm(self.widths*self.skill)
        self.tail=FieldScorer().fit(errors,self.widths);return self
    def all_scores(self,errors):
        e=np.asarray(errors,dtype='f8');z=self.tail.evidence(e)
        return dict(field=e.mean(1),flat=np.average(e,axis=1,weights=self.widths),tail=z.mean(1),tail_flat=np.average(z,axis=1,weights=self.widths),skill_tail=z@self.weights,skill_raw=e@self.weights,skill_equal_tail=z@self.equal,skill_hard_tail=z@self.hard)
    def save(self,path):np.savez_compressed(path,center=self.center,null_error=self.null_error,model_error=self.model_error,widths=self.widths,skill=self.skill,weights=self.weights,equal=self.equal,hard=self.hard,sorted=self.tail.sorted)
