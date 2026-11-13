"""Formula-based, zero-label Audit-HCL financial-schema/budget adaptation.

No device/ownership relation is fabricated. Paths are actual direct payment,
shared payee and shared payer. Training accepts features, edges, activity ONLY.
See protocol for differences from original data/schema and ambiguous paper text.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def path_tables(edges, n, cap=64, seed=0):
    rng = np.random.default_rng(seed)
    forward = [set() for _ in range(n)]; backward = [set() for _ in range(n)]
    for a,b in edges.T:
        forward[a].add(int(b)); backward[b].add(int(a))
    tables = []
    for mode in range(3):
        table = np.full((n, cap), -1, dtype=np.int64)
        for node in range(n):
            if mode == 0:
                neighbors = forward[node] | backward[node]
            elif mode == 1:
                neighbors = set().union(*(backward[v] for v in forward[node])) if forward[node] else set()
            else:
                neighbors = set().union(*(forward[v] for v in backward[node])) if backward[node] else set()
            neighbors.discard(node)
            values = np.array(sorted(neighbors), dtype=np.int64)
            if len(values) > cap: values = rng.choice(values, cap, replace=False)
            table[node, :len(values)] = values
        tables.append(table)
    return np.stack(tables)


class MetaLayer(nn.Module):
    def __init__(self, hidden=128, heads=8):
        super().__init__(); self.hidden, self.heads = hidden, heads
        self.w = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        self.att = nn.Parameter(torch.empty(3, 2, heads, hidden//heads))
        nn.init.xavier_uniform_(self.att)
        self.semantic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.norm = nn.LayerNorm(hidden)

    def attend_chunk(self, h, table, start, stop, relation, augment):
        nb = table[start:stop]; mask = nb >= 0
        # Invalid slots are zero-masked below. Route their (zero) gather
        # gradients to their own rows, avoiding atomic contention on node 0.
        safe = torch.where(mask, nb, torch.arange(start, stop, device=nb.device)[:, None])
        q = h[start:stop].reshape(-1,self.heads,self.hidden//self.heads)
        k = h[safe].reshape(len(nb),nb.shape[1],self.heads,-1)
        logits = F.leaky_relu((q*self.att[relation,0]).sum(-1)[:,None]+(k*self.att[relation,1]).sum(-1),.2)
        if self.training and augment:
            mask = mask & (torch.rand(mask.shape, device=mask.device) >= .3)
        logits = logits.masked_fill(~mask[:,:,None], -1e9)
        weight = logits.softmax(1)*mask[:,:,None]
        weight = weight/weight.sum(1,keepdim=True).clamp_min(1e-8)
        output = (weight[:,:,:,None]*k).sum(1).reshape(len(nb),-1)
        return torch.where(mask.any(1)[:,None], F.elu(output), h[start:stop])

    def forward(self, x, tables, residual=False, augment=False):
        views = []
        for r in range(3):
            h = self.w[r](x); chunks = []
            for start in range(0,len(x),512):
                stop = min(start+512,len(x))
                def call(hh, tt, start=start, stop=stop, relation=r, augment=augment):
                    return self.attend_chunk(hh,tt,start,stop,relation,augment)
                chunks.append(checkpoint(call,h,tables[r],use_reentrant=False) if self.training else call(h,tables[r]))
            views.append(torch.cat(chunks))
        views = torch.stack(views)
        beta = self.semantic(views).mean(1).softmax(0)
        fused = (beta[:,:,None]*views).sum(0)
        return self.norm(fused+x if residual else fused)


class AuditNetwork(nn.Module):
    def __init__(self, features, hidden=128):
        super().__init__(); self.hidden=hidden
        self.project = nn.ModuleList([nn.Linear(features,hidden), nn.Linear(features,hidden)])
        self.layers = nn.ModuleList([MetaLayer(hidden),MetaLayer(hidden)])
        self.position = nn.Parameter(torch.empty(4,hidden)); nn.init.uniform_(self.position,-.01,.01)
        self.gru = nn.GRUCell(hidden,hidden); self.norm=nn.LayerNorm(hidden)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Linear(256,64))
                                    for _ in range(2)])

    def encode(self,x,tables,active,role,augment):
        state=torch.zeros((x.shape[1],self.hidden),device=x.device)
        seen=torch.zeros(x.shape[1],dtype=torch.bool,device=x.device)
        drift=torch.zeros(x.shape[1],device=x.device); count=drift.clone(); last_struct=state.clone()
        for t in range(4):
            values=x[t]
            if augment: values=values*(torch.rand_like(values)>=.2)
            h=torch.where(role[:,None],self.project[0](values),self.project[1](values))
            for i,layer in enumerate(self.layers): h=layer(h,tables[t],residual=i==1,augment=augment)
            new=self.norm(self.gru(h+self.position[t],state))
            continuing=active[t]&seen
            drift=drift+torch.linalg.vector_norm(new-state,dim=1)*continuing
            count=count+continuing
            state=torch.where(active[t,:,None],new,state)
            last_struct=torch.where(active[t,:,None],h,last_struct); seen=seen|active[t]
        return last_struct,state,drift/count.clamp_min(1)

    def views(self,x,tables,active,role,augment):
        if augment:
            # Eq.10 perturbs the structural view; Eq.11 uses the original
            # snapshot sequence for the temporal view. Do not corrupt both.
            s,_,_=self.encode(x,tables,active,role,True)
            _,t,d=self.encode(x,tables,active,role,False)
        else:
            s,t,d=self.encode(x,tables,active,role,False)
        return F.normalize(self.heads[0](s),dim=1), F.normalize(self.heads[1](t),dim=1),d


class AuditHCL:
    def __init__(self,seed=42,device='cpu',steps=200,cap=64):
        self.seed,self.device,self.steps,self.cap=seed,device,steps,cap

    def tensors(self,x,edges,active):
        tables=[torch.as_tensor(path_tables(e,x.shape[1],self.cap,self.seed+t),device=self.device) for t,e in enumerate(edges)]
        return torch.as_tensor(x,device=self.device),tables,torch.as_tensor(active,device=self.device)

    @staticmethod
    def negative_distributions(bank_s,bank_t):
        # The paper holds these distributions fixed within an epoch. Our
        # predeclared 25-update refresh block is the fixed-budget analogue.
        st=torch.softmax(bank_s@bank_t.T/.5,dim=1);st.fill_diagonal_(0)
        ts=torch.softmax(bank_t@bank_s.T/.5,dim=1);ts.fill_diagonal_(0)
        return st,ts

    def fit(self,x,edges,active,role):
        torch.manual_seed(self.seed); self.features=x.shape[2]
        self.model=AuditNetwork(self.features).to(self.device)
        xx,tt,aa=self.tensors(x,edges,active); rr=torch.as_tensor(role,device=self.device)
        opt=torch.optim.AdamW(self.model.parameters(),lr=5e-4,weight_decay=1e-5)
        self.losses=[]; n=x.shape[1]
        self.model.eval()
        with torch.no_grad():
            bank_s,bank_t,_=self.model.views(xx,tt,aa,rr,False)
            negative_st,negative_ts=self.negative_distributions(bank_s,bank_t)
        for step in range(self.steps):
            self.model.train()
            s,t,_=self.model.views(xx,tt,aa,rr,True)
            batch=torch.randperm(n,device=self.device)[:min(256,n)]
            loss=0
            for query,positive,bank,distribution in ((s,t,bank_t,negative_st),(t,s,bank_s,negative_ts)):
                with torch.no_grad():
                    neg=torch.multinomial(distribution[batch],256,replacement=True)
                good=(query[batch]*positive[batch]).sum(1,keepdim=True)/.07
                bad=(query[batch,None]*bank[neg]).sum(-1)/.07
                loss=loss+.5*F.cross_entropy(torch.cat([good,bad],1),torch.zeros(len(batch),dtype=torch.long,device=self.device))
            if not torch.isfinite(loss): raise ValueError('Nonfinite HCL objective')
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),1.); opt.step()
            if (step+1)%25==0:
                self.model.eval()
                with torch.no_grad():
                    bank_s,bank_t,_=self.model.views(xx,tt,aa,rr,False)
                    negative_st,negative_ts=self.negative_distributions(bank_s,bank_t)
            if step%25==0 or step+1==self.steps:
                self.losses.append({'step':step+1,'loss':float(loss.detach().cpu())})
                print(f'audit_hcl step={step+1}/{self.steps} loss={self.losses[-1]["loss"]:.6f}',flush=True)
        self.model.eval()
        self.info=dict(method='audit_hcl_zero_adapted',steps=self.steps,cap=self.cap,hidden=128,heads=8,layers=2,
                       negatives=256,temperature=.07,hard_negative_temperature=.5,mixture=.5,
                       bank_refresh_updates=25,losses=self.losses,supervised_finetune=False,
                       temporal_augmentation=False,negative_distribution_refresh_updates=25,
                       inference='unprojected_structural_temporal_cosine_and_GRU_drift')
        return self

    def score(self,x,edges,active,role):
        xx,tt,aa=self.tensors(x,edges,active); rr=torch.as_tensor(role,device=self.device)
        self.model.eval()
        with torch.inference_mode():
            s,t,d=self.model.encode(xx,tt,aa,rr,False)
            score=.5*(1-F.cosine_similarity(s,t,dim=1))+.5*d
        return score.cpu().numpy().astype(np.float64)

    def save(self,path):
        torch.save(dict(seed=self.seed,steps=self.steps,cap=self.cap,features=self.features,
                        state=self.model.state_dict(),info=self.info),path)

    @classmethod
    def load(cls,path,device='cpu'):
        p=torch.load(path,map_location=device,weights_only=True)
        obj=cls(p['seed'],device,p['steps'],p['cap']);obj.features=p['features'];obj.info=p['info']
        obj.model=AuditNetwork(obj.features).to(device);obj.model.load_state_dict(p['state']);obj.model.eval()
        return obj
