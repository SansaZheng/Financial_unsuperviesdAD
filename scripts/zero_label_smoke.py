"""No financial performance claims: numerical/dependency gates on synthetic graphs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from finad_zero.audit_hcl import AuditHCL
from finad_zero.cae_if import CAEIF
from finad_zero.ebgad import fetch_upstream,score_graph

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
z=np.random.default_rng(43).normal(size=(100,29)).astype('float32')
cae=CAEIF(42,args.device,steps=2).fit(z);cae.save(args.output/'cae')
cs=cae.score(z);cr=CAEIF.load(args.output/'cae',args.device).score(z)
assert cae.info['latent']<z.shape[1] and np.isfinite(cs).all() and np.allclose(cs,cr)
fetch_upstream(args.output/'upstream')
rng=np.random.default_rng(18);n=80
e=np.stack([np.repeat(np.arange(60),3),60+rng.integers(20,size=180)])
x=rng.normal(size=(4,n,10)).astype('float32');role=np.arange(n)<60;x[:,:,0]=role;x[:,:,1]=~role
a=np.ones((4,n),bool)
m=AuditHCL(42,args.device,steps=2).fit(x,[e]*4,a,role);m.save(args.output/'audit.pt')
s=m.score(x,[e]*4,a,role);ss=AuditHCL.load(args.output/'audit.pt',args.device).score(x,[e]*4,a,role)
assert np.isfinite(s).all() and np.allclose(s,ss,rtol=1e-5,atol=1e-6)
flat=x.transpose(1,0,2).reshape(n,-1)
base,info=score_graph(flat,[e]*4,args.output/'spectrum_a')
json.dumps(info,allow_nan=False)  # Undefined author-library diagnostics must be JSON null, never NaN.
perm=rng.permutation(n);inv=np.argsort(perm)
other,other_info=score_graph(flat[perm],[inv[e]]*4,args.output/'spectrum_b')
rho=float(spearmanr(base,other[inv]).statistic)
assert rho>.99999 and info['selected']==other_info['selected'],(rho,info['selected'],other_info['selected'])
(args.output/'smoke.json').write_text(json.dumps({'status':'passed','ebgad_permutation_spearman':rho,
 'ebgad_selected':info['selected'],'audit_reload_max_abs':float(np.max(np.abs(s-ss))),'torch':torch.__version__,
 'cuda':torch.cuda.is_available(),'device':args.device,'cae_latent':cae.info['latent'],
 'cae_reload_max_abs':float(np.max(np.abs(cs-cr)))},indent=2)+'\n')
print('ZERO_LABEL_SMOKE_PASSED',flush=True)
