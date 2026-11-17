import numpy as np
import pytest
import torch
from finad_geometry.models import SkillScorer,TailMap,GeometryHUGE,HIMGeneric,groups
from finad_recent.huge import HUGE,halo

def test_external_teacher_matches_original_training():
    torch.set_num_threads(1);x=np.random.default_rng(0).normal(size=(12,5)).astype('f4');e=np.array([np.arange(12),np.roll(np.arange(12),1)])
    m=HUGE(5,42,steps=2).fit(x,e);n=GeometryHUGE(5,42,steps=2).fit_teacher(x,e,halo(torch.as_tensor(x),torch.as_tensor(e)).numpy())
    for k,v in m.net.state_dict().items():assert torch.equal(v,n.net.state_dict()[k])
    np.testing.assert_array_equal(m.score(x,e),n.score(x,e))

def test_teacher_can_be_independent_of_student_input():
    x=np.ones((12,5),dtype='f4');e=np.array([np.arange(12),np.roll(np.arange(12),1)]);h=np.arange(12,dtype='f4')
    m=GeometryHUGE(5,steps=1).fit_teacher(x,e,h);assert np.isfinite(m.score(x,e)).all()
    with pytest.raises(ValueError):m.fit_teacher(x,e,h[:1])

def test_skill_weights_use_reconstruction_gain_and_width(tmp_path):
    train=np.array([[0.,0.,0.],[1,1,1],[2,2,2]]);cal=np.ones((30,3))*2;errors=np.column_stack([np.ones(30)*4,np.zeros(30)])
    obj=SkillScorer().fit(train,cal,errors,[np.array([0]),np.array([1,2])])
    np.testing.assert_array_equal(obj.skill,[0,1]);np.testing.assert_allclose(obj.weights,[1/9,8/9]);np.testing.assert_allclose(obj.hard,[0,1]);np.testing.assert_allclose(obj.equal,[.2,.8]);obj.save(tmp_path/'s.npz')
    before=obj.weights.copy();obj.all_scores(errors*100);np.testing.assert_array_equal(obj.weights,before)

def test_no_skill_fallback_safe_and_all_constant():
    x=np.zeros((30,3));o=SkillScorer().fit(x,x,np.zeros((30,2)),[np.array([0]),np.array([1,2])]);assert np.isfinite(o.weights).all();np.testing.assert_allclose(o.hard,[1/3,2/3])
    for v in o.all_scores(np.zeros((4,2))).values():assert (v==0).all()

def test_signed_tail_map_and_ties():
    c=np.array([[-2,0],[-2,1],[0,2],[1,3.]])
    obj=TailMap().fit(c);q=np.array([[-3,0],[-2,1],[3,5.]])
    expected=np.array([[-np.log((1+(c[:,j]>=v).sum())/5) for j,v in enumerate(row)] for row in q]);np.testing.assert_array_equal(obj.transform(q),expected)

def test_new_schema_is_complete_and_persistent(tmp_path):
    g=groups('cardsim',24);np.testing.assert_array_equal(np.concatenate(g),np.arange(24));m=HIMGeneric(24,'cardsim',steps=1);m.save(tmp_path/'m.pt');n=HIMGeneric.load(tmp_path/'m.pt');assert n.task=='cardsim' and len(n.groups)==6
    for k,v in m.net.state_dict().items():assert torch.equal(v,n.net.state_dict()[k])
