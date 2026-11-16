"""Synthetic, label-free CPU resource qualification; no financial metrics."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from finad_kronos.stable import VARIANTS,fit_prior,errors,fit_feature_scales,aggregate,thresholds
from finad_kronos.stable_experiment import sources,write

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rng=np.random.default_rng(61116);width=61;n=20000
train=rng.normal(size=(8192,width));prior=fit_prior(train)
observed=rng.normal(size=(n,width));predicted=rng.normal(size=(n,width))*.7
std=np.ones_like(observed);std[::50]=0;counts=np.minimum(32,np.arange(n)%160)
last=rng.normal(size=(n,width));last[counts==0]=prior['center']
started=time.monotonic();checks=[]
for variant in VARIANTS:
    e=errors(observed,predicted,std,counts,last,prior,variant)
    fitted=fit_feature_scales(e,[f'x{i}' for i in range(width)]);s=aggregate(e,fitted);th=thresholds(s)
    assert np.isfinite(s).all() and len(s)==n and th['guarded_pot']['status']=='ok'
    checks.append(dict(variant=variant,finite_scores=len(s),active_dimensions=len(fitted['active_indices'])))
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
result=dict(status='passed',type='synthetic label-free CPU qualification only',financial_metrics=False,
            new_backbone_inference=False,source=sources(),checks=checks,seconds=time.monotonic()-started,
            peak_rss_bytes=rss,synthetic_calibration_rows=n,features=width)
write(a.output/'qualification.json',result);print(json.dumps(result),flush=True)
