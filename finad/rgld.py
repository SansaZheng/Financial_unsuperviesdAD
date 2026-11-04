"""RGLD (arXiv:2606.28970): independent core + train-reference inductive ranks.

No author code is copied. The publication ranks a joint query vector; this
extension freezes all standardizers and empirical rank maps on training data.
"""
import joblib
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from scipy.spatial.distance import cdist


def rank_reference(sorted_training, values):
    """Average-tie normalized ranks, extended by the frozen training ECDF."""
    sorted_training=np.round(sorted_training,10);values=np.round(values,10)
    left=np.searchsorted(sorted_training,values,side='left')
    right=np.searchsorted(sorted_training,values,side='right')
    return np.clip((left+right-1)/(2*max(1,len(sorted_training)-1)),0.,1.)


def local_rms(query,reference,k,query_ids=None,reference_ids=None):
    """Direct squared differences avoid norm-dot cancellation near duplicates.

    Financial categorical/engineered features often create exact duplicates;
    sqrt(||x||²-2x·y+||y||²) can turn zero into 1e-8 and corrupt ECDF ties.
    """
    result=[]
    for i in range(0,len(query),512):
        squared=cdist(query[i:i+512],reference,metric='sqeuclidean')
        if query_ids is not None:
            squared[query_ids[i:i+512,None]==reference_ids[None,:]]=np.inf
        nearest=np.partition(squared,k-1,axis=1)[:,:k]
        result.append(np.sqrt(nearest.mean(1)))
    return np.concatenate(result)


class RGLD:
    def __init__(self,seed=42,global_views=40,local_views=10,random_features=10,references=400):
        self.seed=seed;self.global_views=global_views;self.local_views=local_views
        self.random_features=random_features;self.references=references

    def _view(self,p,rng):
        r=2 if p<=3 else 5 if p<=20 else 8 if p<=40 else 12 if p<=100 else 16
        columns=np.sort(rng.choice(p,min(p,max(3,int(np.ceil(.3*p)))),replace=False))
        projection=rng.normal(size=(len(columns),r))/np.sqrt(r)
        return columns,projection

    @staticmethod
    def _global_raw(x,view):
        columns,projection,w,phase,bandwidth,means,_,_,_=view
        projected=(x[:,columns].astype(np.float64)@projection)@w
        scale=np.sqrt(2/w.shape[1])
        return np.column_stack([-(scale*np.cos(projected/s+phase))@mu
                                for s,mu in zip(bandwidth,means)])

    def fit(self,x):
        x=np.asarray(x,dtype=np.float32);rng=np.random.default_rng(self.seed)
        if len(x)<2 or x.ndim!=2:raise ValueError('RGLD requires >=2 finite training rows')
        self.globals=[];self.locals=[]
        global_rank=[];local_rank=[]
        for _ in range(self.global_views):
            columns,projection=self._view(x.shape[1],rng);z=x[:,columns].astype(np.float64)@projection
            pairs=rng.integers(0,len(x),size=(8192,2));dist=np.linalg.norm(z[pairs[:,0]]-z[pairs[:,1]],axis=1)
            bandwidth=float(np.median(dist));bandwidth=bandwidth if bandwidth>1e-8 else 1.
            sampler=qmc.Sobol(z.shape[1],scramble=True,seed=int(rng.integers(2**31)))
            uniform=sampler.random_base2(int(np.ceil(np.log2(self.random_features))))[:self.random_features]
            w=ndtri(np.clip(uniform,1e-7,1-1e-7)).T
            phase=rng.uniform(0,2*np.pi,self.random_features)
            scales=bandwidth*np.array([.25,.5,1.,2.]);projected=z.astype(np.float64)@w
            maps=[np.sqrt(2/self.random_features)*np.cos(projected/s+phase) for s in scales]
            means=np.array([f.mean(0) for f in maps]);raw=np.column_stack([-f@mu for f,mu in zip(maps,means)])
            center=raw.mean(0);std=raw.std(0);std[std<1e-12]=1.
            score=((raw-center)/std).mean(1);order=np.sort(score)
            self.globals.append((columns,projection,w,phase,scales,means,center,std,order))
            global_rank.append(rank_reference(order,score))
        for _ in range(self.local_views):
            columns,projection=self._view(x.shape[1],rng);z=x[:,columns].astype(np.float64)@projection
            ids=np.sort(rng.choice(len(x),min(self.references,len(x)),replace=False))
            # A reference training row must not become its own nearest neighbor.
            k=min(10,len(ids)-1);reference=z[ids].copy()
            score=local_rms(z,reference,k,np.arange(len(x)),ids);order=np.sort(score)
            self.locals.append((columns,projection,reference,k,order))
            local_rank.append(rank_reference(order,score))
        g=np.mean(global_rank,axis=0);l=np.mean(local_rank,axis=0)
        self.branch_reference=(np.sort(g),np.sort(l))
        self.info=dict(method='rgld2026',source='arXiv:2606.28970v1',variant='train_reference_inductive',
                       global_views=self.global_views,local_views=self.local_views,random_features=self.random_features,
                       reference_cap=self.references,neighbors=10,qmc='scrambled_Sobol_inverse_normal',
                       bandwidth_multipliers=[.25,.5,1.,2.],rank_fit='training_only_average_ties',
                       local_reference_training_self_excluded=True,preprocessing='shared_v1_robust_clipped')
        self.info['local_distance_backend']='float64_direct_squared_differences_cdist'
        return self

    def score(self,x):
        g=np.zeros(len(x));l=np.zeros(len(x))
        for view in self.globals:
            raw=self._global_raw(x,view);center,std,reference=view[-3:]
            g+=rank_reference(reference,((raw-center)/std).mean(1))/len(self.globals)
        for columns,projection,points,k,reference in self.locals:
            score=local_rms(x[:,columns].astype(np.float64)@projection,points,k)
            l+=rank_reference(reference,score)/len(self.locals)
        return .5*(rank_reference(self.branch_reference[0],g)+rank_reference(self.branch_reference[1],l))

    def save(self,path):joblib.dump(self,path,compress=3)

    @classmethod
    def load(cls,path):return joblib.load(path)  # Own/trusted checkpoints only.
