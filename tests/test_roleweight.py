import numpy as np
import pytest
from finad_roleweight.adapters import RoleAdapter,VARIANTS
from finad_roleweight.fields import FieldScorer
from finad_repair.roles import RoleNormalizer
from finad.preprocessing import Preprocessor

def fixture():
    x=np.random.default_rng(7).normal(size=(4,40,10)).astype('f4');r=np.arange(40)%2==0
    x[:,:,0]=r;x[:,:,1]=~r;x[:,~r,2::2]=0;x[:,r,3::2]=0
    return x,r

@pytest.mark.parametrize('mode',VARIANTS)
def test_role_roundtrip_and_width(mode,tmp_path):
    x,r=fixture();o=RoleAdapter(mode).fit(x,r);o.save(tmp_path)
    a=o.transform(x,r);np.testing.assert_array_equal(a,RoleAdapter.load(tmp_path).transform(x,r))
    assert a.shape==(40,40 if mode=='legacy' else 32 if mode=='numeric' else 48 if mode.startswith('dual') else 16)

def test_anchor_and_original_channel():
    x,r=fixture();old=Preprocessor().fit(x.reshape(-1,10)).transform(x.reshape(-1,10)).reshape(x.shape).transpose(1,0,2).reshape(40,40)
    np.testing.assert_array_equal(RoleAdapter('legacy').fit(x,r).transform(x,r),old)
    np.testing.assert_array_equal(RoleAdapter('role').fit(x,r).transform(x,r),RoleNormalizer().fit(x,r).transform(x,r))
    numeric=RoleAdapter('numeric').fit(x,r).transform(x,r)
    np.testing.assert_array_equal(numeric,old.reshape(40,4,10)[:,:,2:].reshape(40,32))
    dual=RoleAdapter('dual').fit(x,r).transform(x,r)
    np.testing.assert_allclose(dual[:,:32]*np.sqrt(32),numeric,atol=5e-7,rtol=1e-7)

def test_roles_validation():
    x,r=fixture()
    with pytest.raises(ValueError):RoleAdapter('dual').fit(x,np.ones(40,dtype=bool))
    with pytest.raises(ValueError):RoleAdapter('bad')

def test_empirical_tail_ties_and_bounds(tmp_path):
    c=np.array([[0.,1],[0,2],[2,2],[3,4]])
    o=FieldScorer().fit(c,[1,8]);q=np.array([[0,0],[1,2],[100,100]])
    expected=np.array([[-np.log((1+(c[:,j]>=v).sum())/5) for j,v in enumerate(row)] for row in q])
    np.testing.assert_array_equal(o.evidence(q),expected)
    assert np.max(expected)==np.log(5)
    o.save(tmp_path/'f.npz');loaded=FieldScorer.load(tmp_path/'f.npz')
    for k,v in o.all_scores(q).items():np.testing.assert_array_equal(v,loaded.all_scores(q)[k])

def test_scale_invariance_and_frozen_fit():
    c=np.random.default_rng(0).uniform(.1,8,(100,4));t=c[:10]*2
    a=FieldScorer().fit(c,[1,4,8,16]);b=FieldScorer().fit(c*[2,10,.1,30],[1,4,8,16])
    for k in ['mean_scale','median_scale','tail','tail_corr']:
        np.testing.assert_allclose(a.all_scores(t)[k],b.all_scores(t*[2,10,.1,30])[k],atol=1e-14)
    before=a.sorted.copy();a.all_scores(t*100);np.testing.assert_array_equal(before,a.sorted)

def test_constant_fields_are_safe():
    c=np.zeros((50,3));o=FieldScorer().fit(c,[1,2,3])
    for v in o.all_scores(c).values():assert np.isfinite(v).all();assert (v==0).all()
    np.testing.assert_allclose(o.corr_weights,np.ones(3)/3)
    assert o.evidence([[1,0,0]])[0,0]==np.log(51)
    assert FieldScorer().fit(c[:,:1],[1]).corr_weights[0]==1

@pytest.mark.parametrize('x',[np.array([1]),np.array([[np.nan]]),np.array([[-1.]])])
def test_invalid_field_errors(x):
    with pytest.raises(ValueError):FieldScorer().fit(x,[1])
