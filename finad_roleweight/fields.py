"""Unlabeled calibration-error transforms, fixed before new test evaluation."""
import numpy as np
VARIANTS=('field','flat','mean_scale','median_scale','tail','tail_flat','tail_max','tail_corr')

class FieldScorer:
    @staticmethod
    def validate(x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all() or (x<0).any(): raise ValueError('Finite nonnegative NxG errors required')
        return x

    def fit(self, calibration, widths):
        x=self.validate(calibration);self.widths=np.asarray(widths,dtype=np.float64)
        if self.widths.shape!=(x.shape[1],) or not np.isfinite(self.widths).all() or (self.widths<=0).any():raise ValueError('Invalid field widths')
        self.sorted=np.sort(x,axis=0);mean=x.mean(0);median=np.median(x,axis=0)
        self.mean=np.where(mean>1e-8,mean,1.)
        self.median=np.where(median>1e-8,median,self.mean)
        z=self.evidence(x);g=x.shape[1]
        if g==1:corr=np.zeros((1,1))
        else:
            centered=z-z.mean(0);cov=centered.T@centered;norm=np.sqrt(np.diag(cov));denom=norm[:,None]*norm[None,:]
            corr=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
        np.fill_diagonal(corr,0);w=1/(1+np.clip(corr,0,1).sum(1));self.corr_weights=w/w.sum()
        return self

    def evidence(self,x):
        x=self.validate(x)
        if x.shape[1]!=self.sorted.shape[1]:raise ValueError('Field mismatch')
        n=len(self.sorted)
        p=np.column_stack([(1+n-np.searchsorted(self.sorted[:,g],x[:,g],side='left'))/(n+1) for g in range(x.shape[1])])
        return -np.log(p)

    def all_scores(self,x):
        x=self.validate(x);z=self.evidence(x)
        return dict(field=x.mean(1),flat=np.average(x,axis=1,weights=self.widths),mean_scale=(x/self.mean).mean(1),median_scale=(x/self.median).mean(1),tail=z.mean(1),tail_flat=np.average(z,axis=1,weights=self.widths),tail_max=z.max(1),tail_corr=z@self.corr_weights)

    def save(self,path):np.savez_compressed(path,sorted=self.sorted,widths=self.widths,mean=self.mean,median=self.median,corr_weights=self.corr_weights)

    @classmethod
    def load(cls,path):
        o=cls()
        with np.load(path,allow_pickle=False) as z:
            for k in z.files:setattr(o,k,z[k])
        return o
