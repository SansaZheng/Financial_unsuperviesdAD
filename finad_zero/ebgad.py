"""Thin label-free wrapper of pinned author code, downloaded outside this repo.

The released mirror-null selector differs from paper Appendix B.4's ordered
regime selector. We reproduce the RELEASED variant, not claim exact paper scores.
No upstream source is redistributed (repository has no license file).
"""
import hashlib
import io
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import urllib.request
import numpy as np
import torch
from scipy.stats import rankdata

COMMIT = 'cba2969e390ae6cd4381fe86393fa6cfcfc40d13'
TAR_SHA256 = 'f1a43e1136212d63ec82dcb6084d32ea00d8720e99987e0b6f2300b754b19f2c'


def fetch_upstream(root):
    root=Path(root); archive=root/'ebgad.tar.gz'; dest=root/COMMIT
    root.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        raw=urllib.request.urlopen(f'https://codeload.github.com/heraclixus/EBGAD/tar.gz/{COMMIT}',timeout=120).read()
        if hashlib.sha256(raw).hexdigest()!=TAR_SHA256: raise ValueError('Upstream archive SHA mismatch')
        archive.write_bytes(raw)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=TAR_SHA256: raise ValueError('Upstream archive SHA mismatch')
    if not dest.exists():
        dest.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                relative=Path(*Path(member.name).parts[1:])
                if not relative.parts: continue
                if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                    raise ValueError('Unsafe upstream archive path')
                target=dest/relative
                if member.isdir(): target.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(tar.extractfile(member).read())
    sys.path.insert(0,str(dest))
    return dest


def score_graph(x,edges,cache):
    """Transductive per-phase empirical Bayes; no target argument or target access."""
    from ebgad.prep import prepare
    from ebgad.fit import fit_prepared
    from ebgad.scores import bank_specs, profile_specs, compute_members, conditional_residual
    from ebgad.nulls import member_tails
    from ebgad.select import select
    from ebgad.graphstats import compute_graph_stats, density_adjusted_template_gammas
    both=np.unique(np.concatenate([np.concatenate(edges,1),np.concatenate(edges,1)[::-1]],1),axis=1)
    data=SimpleNamespace(x=torch.as_tensor(np.asarray(x,dtype=np.float32)),edge_index=torch.as_tensor(both),num_nodes=len(x))
    h,d=compute_graph_stats(data); gammas=density_adjusted_template_gammas(h,d)
    spectrum={}; best=None; fit=None
    for gamma in gammas:
        prep=prepare('financial',data,gamma,pca=None,pca_order='paper',force_full=True,seed=0,
                     cache_dir=str(cache),_spectrum_cache=spectrum)
        f=fit_prepared(prep)
        if fit is None or f.ml>fit.ml: best,fit=prep,f
    specs=bank_specs(fit.q,[(np.inf,np.inf)])+profile_specs(fit.q)
    members=compute_members(best,fit.q,specs,start='template',rough='isotropic')
    members.append(conditional_residual(best,fit.rho,fit.kappa))
    # Same rank-duplicate policy as author's entry point; priority J*,R,P.
    members=sorted(enumerate(members),key=lambda p:(0 if p[1].name in ('J*','R','P','NP') else 1,p[0]))
    ranks=[]; pool={}; raw={}
    for _,m in members:
        r=rankdata(m.score); r=(r-r.mean())/max(r.std(),1e-12)
        if any(float(np.mean(r*previous))>=.999 for previous in ranks): continue
        ranks.append(r); entry={'stat':m.score,'null_mean':m.null_mean(best.d)}
        try: entry['sf'],entry['cdf']=member_tails(m,best.d)
        except NotImplementedError: pass
        pool[m.name]=entry; raw[m.name]=m.score
    selected,candidates=select(pool)
    if selected is None: raise ValueError('All EB-GAD candidates degenerate')
    score=np.asarray(raw[selected],dtype=np.float64)
    if not np.isfinite(score).all(): raise ValueError('Nonfinite EB-GAD score')
    info={'method':'ebgad_release_zero','upstream_commit':COMMIT,'selected':selected,
          'gamma_profile':gammas,'fit':fit.as_dict(),'nodes':len(x),'spectrum':'full float64',
          'transductive':True,'candidates':{k:{field:(None if isinstance(value,float) and not np.isfinite(value) else value)
              for field,value in v.summary().items()} for k,v in candidates.items()},
          'score':'raw selected statistic, matching author entry point; no local-fdr replacement'}
    return score,info
