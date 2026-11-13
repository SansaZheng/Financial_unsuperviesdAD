import inspect
import json
import numpy as np
import torch
from finad_zero.cae_if import CAEIF
from finad_zero.audit_hcl import AuditHCL, path_tables
from finad_zero.graph_data import aggregate_targets, make_phase


def toy_graph(n=12):
    src=np.repeat(np.arange(n),2);dst=n+(src+np.tile([0,1],n))%5
    edges=np.stack([src,dst]); rng=np.random.default_rng(23)
    x=rng.normal(size=(4,n+5,10)).astype('float32');x[:,:,0]=0;x[:,:n,0]=1
    return x,[edges]*4,np.ones((4,n+5),bool),np.arange(n+5)<n


def test_label_blind_interfaces():
    for function in (CAEIF.fit,AuditHCL.fit,make_phase,path_tables):
        assert not {'y','labels','targets'}&set(inspect.signature(function).parameters)


def test_source_archive_requires_pinned_sha(monkeypatch):
    import subprocess
    import pytest
    from finad_zero.experiment import code_commit
    def no_git(*args,**kwargs):raise subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(subprocess,'check_output',no_git);monkeypatch.delenv('FINAD_CODE_COMMIT',raising=False)
    with pytest.raises(ValueError,match='requires FINAD_CODE_COMMIT'):code_commit()
    monkeypatch.setenv('FINAD_CODE_COMMIT','a'*40);assert code_commit()=='a'*40
    monkeypatch.setenv('FINAD_CODE_COMMIT','truncated')
    with pytest.raises(ValueError,match='complete lowercase'):code_commit()


def test_graph_features_targets_separated():
    a,b=np.array(['a','a','b']),np.array(['c','d','c'])
    first=make_phase(a,b,np.arange(3),np.arange(3.))
    second=make_phase(a,b,np.arange(3),np.arange(3.))
    np.testing.assert_array_equal(first[0],second[0])
    y=aggregate_targets(np.array([0,0,1]),np.array([2,3,2]),np.array([0,-1,1]),4)
    np.testing.assert_array_equal(y,[-1,1,1,-1])


def test_cae_gradient_reload_chunk(tmp_path):
    torch.set_num_threads(2);z=np.random.default_rng(5).normal(size=(80,11)).astype('float32')
    m=CAEIF(42,steps=3).fit(z);m.save(tmp_path/'m');loaded=CAEIF.load(tmp_path/'m')
    assert m.losses[-1]['loss']>0
    assert m.info['latent'] < z.shape[1]  # Compact representation on financial schemas (d >= 7).
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in m.model.parameters())
    np.testing.assert_allclose(m.score(z),loaded.score(z),rtol=1e-7)
    np.testing.assert_allclose(m.score(z),np.r_[m.score(z[:7]),m.score(z[7:])],rtol=1e-7)


def test_audit_gradient_reload_permutation(tmp_path):
    torch.set_num_threads(2);x,e,a,r=toy_graph();m=AuditHCL(42,steps=2).fit(x,e,a,r)
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in m.model.layers.parameters())
    for block in (m.model.gru,m.model.heads):
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in block.parameters())
    m.save(tmp_path/'m.pt');loaded=AuditHCL.load(tmp_path/'m.pt')
    score=m.score(x,e,a,r);np.testing.assert_allclose(score,loaded.score(x,e,a,r),rtol=1e-6)
    xx,tt,aa=m.tensors(x,e,a);rr=torch.as_tensor(r);m.model.train()
    with torch.no_grad():
        clean_a=m.model.encode(xx,tt,aa,rr,False)
        torch.manual_seed(919);clean_b=m.model.encode(xx,tt,aa,rr,False)
    for first,second in zip(clean_a,clean_b):torch.testing.assert_close(first,second,rtol=0,atol=0)
    m.model.eval()
    perm=np.random.default_rng(7).permutation(x.shape[1]);inverse=np.argsort(perm)
    ps=m.score(x[:,perm],[inverse[ee] for ee in e],a[:,perm],r[perm])
    np.testing.assert_allclose(score,ps[inverse],rtol=1e-4,atol=2e-5)


def test_no_fake_paths():
    tables=path_tables(np.array([[0,1],[2,2]]),4,cap=5)
    assert 2 in tables[0,0] and 1 in tables[1,0]
    assert (tables[:,3]==-1).all()
    assert (tables[2,:2]==-1).all()


def test_runner_needs_no_training_or_calibration_labels(tmp_path,monkeypatch):
    from finad_zero.experiment import tabular_run
    data=tmp_path/'data';out=tmp_path/'out';data.mkdir();out.mkdir()
    rng=np.random.default_rng(5)
    for split in ('train','calibration','test'):
        rng.normal(size=(50,7)).astype('float32').tofile(data/f'{split}.x.f32')
    np.tile([0,1],25).astype('int8').tofile(data/'test.y.i8')
    (data/'manifest.json').write_text(json.dumps(dict(features=list('abcdefg'),splits={s:dict(rows=50) for s in ('train','calibration','test')})))
    real=np.fromfile;reads=[]
    def checked(path,*a,**kw):
        reads.append(str(path));assert (out/'scores.npz').exists() and (out/'thresholds.json').exists()
        return real(path,*a,**kw)
    monkeypatch.setattr(np,'fromfile',checked)
    result,_,_=tabular_run(data,out,'cae_if_core',42,'cpu',2)
    assert reads==[str(data/'test.y.i8')] and result['evaluated']==50
