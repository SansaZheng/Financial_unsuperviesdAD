"""RTTAD core (arXiv:2605.10242), bounded transductive financial adaptation.

Independent implementation of learned masks, dual reconstruction, signed
adaptation, GammaGMM confidence and kNN embedding contrast. Concrete choices
missing from the paper and finite budgets are specified in docs/METHODS_2026_CN.md.
"""
import copy
import warnings
import numpy as np
import torch
from torch import nn


class MaskedDualNetwork(nn.Module):
    def __init__(self,features,masks=4,hidden=128,latent=32):
        super().__init__();self.features=features;self.masks=masks
        self.generator=nn.Sequential(nn.Linear(features,hidden),nn.ReLU(),nn.Linear(hidden,masks*features))
        self.encoder=nn.Sequential(nn.Linear(features,hidden),nn.ReLU(),nn.Linear(hidden,latent))
        self.decoder=nn.Sequential(nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,features))
        self.auxiliary=nn.Sequential(nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,latent))

    def mask_errors(self,x):
        masks=self.generator(x).reshape(len(x),self.masks,self.features).sigmoid()
        masked=self.encoder(x[:,None,:]*masks)
        return (self.decoder(masked)-x[:,None,:]).square().sum(-1)

    def terms(self,x):
        masks=self.generator(x).reshape(len(x),self.masks,self.features).sigmoid()
        embedding=self.encoder(x);masked=self.encoder(x[:,None,:]*masks)
        reconstruction=(self.decoder(masked)-x[:,None,:]).square().sum(-1).mean(-1)
        # Both encodings participate in the auxiliary task, as in equation (6).
        auxiliary=(self.auxiliary(masked)-embedding[:,None,:]).square().sum(-1).mean(-1)
        similarity=masks@masks.transpose(1,2)
        diagonal=torch.eye(self.masks,dtype=torch.bool,device=x.device)
        diversity=similarity.masked_fill(diagonal,float('-inf')).logsumexp(-1).sum(-1)/self.features
        return reconstruction,auxiliary,diversity,embedding


def gamma_confidence(scores,seed):
    """Use actual PyThresh GammaGMM, not an ordinary two-component GMM proxy."""
    from pythresh.thresholds.gamgmm import GAMGMM
    class CapturePosterior(GAMGMM):
        def _compute_gamma_posterior(self,decision):
            self.posterior_=super()._compute_gamma_posterior(decision)
            return self.posterior_
    scores=np.asarray(scores,dtype=np.float64)
    scalar=scores.mean(1) if scores.ndim==2 else scores
    if len(scores)<20 or np.ptp(scores)<1e-12:
        return None,dict(status='insufficient_or_constant_scores')
    state=np.random.get_state()
    estimator=CapturePosterior(K=min(10,len(scores)-1),n_contaminations=512,n_draws=25,
                              steps=2,random_state=int(seed),skip=False)
    # GammaGMM uses NumPy's global RNG; restore it so selection has no side effect.
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always');estimator.eval(scores)
        posterior=np.asarray(estimator.posterior_)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
        return None,dict(status='gamma_failure',reason=str(error))
    finally:np.random.set_state(state)
    if not np.isfinite(posterior).all():return None,dict(status='nonfinite_gamma_posterior')
    info=dict(status='ok',warnings=sorted(set(str(w.message) for w in captured)),
              contamination_mean=float(posterior.mean()),components_cap=10,posterior_draws=512)
    if not np.isfinite(posterior).all() or not np.any(posterior>0):
        return None,{**info,'status':'no_positive_gamma_posterior'}
    low,high=np.quantile(posterior,[.05,.95]);normal=np.quantile(scalar,1-high);anomaly=np.quantile(scalar,1-low)
    # Separate strict tails; tied/ambiguous observations never get both signs.
    normal_ids=np.flatnonzero(scalar<normal);anomaly_ids=np.flatnonzero(scalar>anomaly)
    info.update(contamination_interval_90=[float(low),float(high)],normal_threshold=float(normal),
                anomaly_threshold=float(anomaly),pseudo_normal=len(normal_ids),pseudo_anomaly=len(anomaly_ids))
    return (normal_ids,anomaly_ids),info


class RTTAD:
    def __init__(self,seed=42,device='cpu',steps=400,adapt_rounds=3,adapt_steps=10,masks=4):
        self.seed=seed;self.device=device;self.steps=steps;self.adapt_rounds=adapt_rounds
        self.adapt_steps=adapt_steps;self.masks=masks

    @staticmethod
    def task_loss(terms):
        main,aux,div,_=terms;weight=torch.minimum(main.mean().detach().new_tensor(1.),1/main.mean().detach().clamp_min(1e-8))
        return main+weight*aux+.01*div

    def fit(self,x):
        torch.manual_seed(self.seed);x=np.asarray(x,dtype=np.float32)
        self.features=x.shape[1];self.latent=min(32,max(2,self.features//2))
        self.model=MaskedDualNetwork(self.features,self.masks,latent=self.latent).to(self.device)
        rng=np.random.default_rng(self.seed);optimizer=torch.optim.Adam(self.model.parameters(),lr=.001,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.98);epoch_steps=max(1,int(np.ceil(len(x)/256)))
        self.losses=[]
        for step in range(self.steps):
            batch=torch.as_tensor(x[rng.integers(len(x),size=min(256,len(x)))],device=self.device)
            loss=self.task_loss(self.model.terms(batch)).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite RTTAD training objective')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if (step+1)%epoch_steps==0:scheduler.step()
            if step%100==0 or step==self.steps-1:self.losses.append(dict(step=step+1,loss=float(loss.detach().cpu())))
        self.model.eval();self.training_anchor=x[rng.choice(len(x),min(1024,len(x)),replace=False)].copy()
        self.info=dict(method='rttad2026',source='arXiv:2605.10242v1',variant='bounded_transductive_financial',
                       features=self.features,latent=self.latent,masks=self.masks,steps=self.steps,batch_size=256,
                       lr=.001,weight_decay=1e-5,epoch_lr_decay=.98,gamma=.01,temperature=1.,diversity_scale='1/features',
                       score='mean_masked_squared_reconstruction_sum',training_anchor_cap=1024,
                       training_labels_used=False,adaptation_labels_used=False,losses=self.losses,adaptation=[])
        return self

    def score(self,x):
        if len(x)>512:
            return np.concatenate([self.score(x[i:i+512]) for i in range(0,len(x),512)])
        self.model.eval()
        with torch.inference_mode():
            return self.model.mask_errors(torch.as_tensor(x,dtype=torch.float32,device=self.device)).mean(1).cpu().numpy().astype(np.float64)

    def component_scores(self,x):
        self.model.eval();out=[]
        with torch.inference_mode():
            for i in range(0,len(x),512):
                errors=self.model.mask_errors(torch.as_tensor(x[i:i+512],dtype=torch.float32,device=self.device))
                out.append(errors.cpu().numpy().astype(np.float64))
        return np.concatenate(out)

    def _scores(self,x):
        return np.concatenate([self.score(x[i:i+512]) for i in range(0,len(x),512)])

    def adapt(self,unlabelled):
        x=np.asarray(unlabelled,dtype=np.float32);rng=np.random.default_rng(self.seed+9000)
        anchor=self.training_anchor.copy()
        for round_id in range(self.adapt_rounds):
            chosen,selection=gamma_confidence(self.component_scores(x),self.seed+round_id)
            record=dict(round=round_id+1,selection=selection)
            if chosen is None:
                self.info['adaptation'].append(record);break
            normal,anomaly=chosen
            if len(normal)<2:
                record['status']='not_enough_pseudo_normal';self.info['adaptation'].append(record);break
            before=copy.deepcopy(self.model.state_dict())
            with torch.no_grad():
                anchor_tensor=torch.as_tensor(anchor,device=self.device)
                normal_anchor=self.model.encoder(anchor_tensor).detach()
            optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-5)
            for step in range(self.adapt_steps):
                signed=[]
                for ids,sign in ((normal,1.),(anomaly,-1.)):
                    if not len(ids):continue
                    batch=torch.as_tensor(x[rng.choice(ids,min(128,len(ids)),replace=False)],device=self.device)
                    terms=self.model.terms(batch);embedding=terms[-1]
                    # Distances used to SELECT neighbors are detached. Gradients
                    # flow through the current query embedding, not fixed anchors.
                    with torch.no_grad():
                        distance=(embedding.detach()[:,None,:]-normal_anchor[None,:,:]).square().sum(-1)
                        nearest=distance.topk(min(3,len(normal_anchor)),largest=False).indices
                        target=normal_anchor[nearest]
                    contrast=(embedding[:,None,:]-target).square().sum(-1).mean(-1)
                    signed.append(sign*(self.task_loss(terms)+contrast).mean())
                loss=sum(signed)
                if not torch.isfinite(loss):
                    self.model.load_state_dict(before);record['status']='nonfinite_round_rolled_back';break
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            else:record['status']='updated'
            self.model.eval();record.update(gradient_steps=step+1,normal_selected=len(normal),anomaly_selected=len(anomaly))
            self.info['adaptation'].append(record)
            if record['status']!='updated':break
            # The normal pool grows from pseudo-labels, never from target files.
            joined=np.concatenate([anchor,x[normal]]);anchor=joined[rng.choice(len(joined),min(1024,len(joined)),replace=False)]
        self.info.update(adaptation_pool_rows=len(x),adaptation_round_cap=self.adapt_rounds,
                         adaptation_steps_per_round=self.adapt_steps,inference_state='frozen_after_test_pool_adaptation')
        return self

    def save(self,path):
        torch.save(dict(seed=self.seed,features=self.features,latent=self.latent,masks=self.masks,steps=self.steps,
                        adapt_rounds=self.adapt_rounds,adapt_steps=self.adapt_steps,info=self.info,
                        state={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                        anchor=torch.as_tensor(self.training_anchor)),path)

    @classmethod
    def load(cls,path,device='cpu'):
        p=torch.load(path,map_location='cpu',weights_only=True)
        obj=cls(p['seed'],device,p['steps'],p['adapt_rounds'],p['adapt_steps'],p['masks'])
        obj.features=p['features'];obj.latent=p['latent'];obj.info=p['info'];obj.training_anchor=p['anchor'].numpy()
        obj.model=MaskedDualNetwork(obj.features,obj.masks,latent=obj.latent).to(device)
        obj.model.load_state_dict(p['state']);obj.model.eval();return obj
