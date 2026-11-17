"""Counterexamples to legacy adaptations and independent repair invariants."""
import copy
import numpy as np
import pytest
import torch
from finad.preprocessing import Preprocessor
from finad_recent.himvh import hyperedges, discrepancies, HimvhNet
from finad_recent.huge import halo, loop_edges, similarity
from finad_repair.hypergraph import NativeContext, block_forward, assignments, masked
from finad_repair.roles import RoleNormalizer, canonical, score_without_self


def test_native_ego_matches_full_forward_and_parameter_gradients():
    torch.set_num_threads(2);torch.manual_seed(3)
    x=np.random.default_rng(3).normal(size=(20,3)).astype(np.float32)
    views=[hyperedges(np.arange(20)//5),hyperedges(np.arange(20)%4)]*2
    ctx=NativeContext(x,views);enh=ctx.enhanced(x,np.arange(20))
    np.testing.assert_allclose(enh,discrepancies(x,views),atol=1e-7)
    targets=np.array([7,1,14]);net=HimvhNet(3,4,hidden=8,layers=3).eval();other=copy.deepcopy(net)
    dense=net(torch.tensor(enh),[torch.tensor(e) for e in views])[targets]
    plan=ctx.plan(targets);needed=np.unique(np.concatenate([b[0][0] for b in plan]))
    block=block_forward(other,ctx.enhanced(x,needed),plan,targets,'cpu',needed)
    torch.testing.assert_close(dense,block,rtol=1e-5,atol=1e-6)
    dense.square().sum().backward();block.square().sum().backward()
    for (name,a),(n,b) in zip(net.named_parameters(),other.named_parameters()):
        assert name==n;torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)


def test_legacy_clean_auxiliary_leaks_masked_value_corrected_aux_does_not():
    x=np.random.default_rng(8).normal(size=(12,3)).astype(np.float32);changed=x.copy();changed[0,0]+=20
    views=[hyperedges(np.arange(12)//4),hyperedges(np.arange(12)%3)]*2
    old_a=discrepancies(x,views);old_b=discrepancies(changed,views)
    old_a[0,0]=old_b[0,0]=0
    assert np.max(np.abs(old_a-old_b))>1e-4  # hidden value remains in clean auxiliary channels
    x[0,0]=changed[0,0]=0
    c=NativeContext(x,views)
    np.testing.assert_array_equal(c.enhanced(x,np.arange(12)),c.enhanced(changed,np.arange(12)))


def test_role_semantics_alignment_removes_artificial_heterophily():
    raw=np.zeros((4,4,10),np.float32);role=np.array([1,1,0,0],bool)
    raw[:,:,0]=role;raw[:,:,1]=~role
    raw[:,role,2]=2;raw[:,role,4]=3;raw[:,role,6]=1;raw[:,role,8]=4
    raw[:,~role,3]=2;raw[:,~role,5]=3;raw[:,~role,7]=1;raw[:,~role,9]=4
    e=torch.tensor([[0,2,1,3],[2,0,3,1]])
    original=torch.tensor(raw.transpose(1,0,2).reshape(4,-1))
    assert torch.min(halo(original,e))>0
    normalized=RoleNormalizer().fit(raw,role).transform(raw,role)
    np.testing.assert_array_equal(normalized,np.zeros((4,16)))
    torch.testing.assert_close(halo(torch.tensor(normalized),e),torch.zeros(4))


def test_no_self_score_is_exact_neighbor_cosine():
    from types import SimpleNamespace
    z=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    e=torch.tensor([[0,1,1,2],[1,0,2,1]])
    model=SimpleNamespace(device='cpu',net=torch.nn.Module());model.net.mlp=torch.nn.Identity()
    for values in [z,torch.zeros_like(z)]:
        direct=-similarity(values,e).numpy()
        np.testing.assert_allclose(score_without_self(model,values.numpy(),e.numpy()),direct,atol=1e-7)


def test_stateless_masks_preserve_node_identity_and_cover_every_field():
    ids=np.array([4,77,9,12]);a=assignments(ids,42,5)
    np.testing.assert_array_equal(a[[2,0]],assignments(ids[[2,0]],42,5))
    for i in range(4):assert set((a[i]+np.arange(5))%5)==set(range(5))


def test_endpoint_bce_removes_160_fold_endpoint_gradient_dilution():
    from torch.nn import functional as F
    a=torch.zeros((2,160),requires_grad=True);target=torch.zeros_like(a);target[1,-13:]=1
    F.binary_cross_entropy_with_logits(a,target).backward();full=a.grad[:,-1].clone();a.grad=None
    F.binary_cross_entropy_with_logits(a[:,-1],target[:,-1]).backward()
    torch.testing.assert_close(a.grad[:,-1],full*160)


def test_rebuilt_market_statistics_match_original_formulas():
    pytest.importorskip('pywt')
    from finad_repair.sdfmm import point_features
    from scripts.prepare_repo_data import point_features as reference
    value=np.random.default_rng(2).uniform(.1,1,160).astype(np.float32)
    np.testing.assert_allclose(point_features(value),reference(value[None,:,None])[0],rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('task,width', [('sapienza_15S',12),('sd_stock',11)])
def test_coherent_generator_preserves_clocks_order_and_derived_features(task,width):
    pytest.importorskip('pywt')
    from finad_repair.sdfmm import coherent,point_features
    rng=np.random.default_rng(4)
    if width==12:
        x=np.ones((2,160,12),np.float32);x[:,:,6]=3;x[:,:,7]=4
        phase=np.linspace(0,6,160);x[:,:,8]=np.sin(phase);x[:,:,9]=np.cos(phase);x[:,:,10:]=x[:,:,8:10]
    else:x=np.stack([point_features(rng.uniform(.1,1,160)) for _ in range(2)])
    changed=coherent(x,np.array([300,300]),task,rng)
    np.testing.assert_array_equal(changed[:,:-13],x[:,:-13]);assert np.max(np.abs(changed-x))>0
    if width==12:
        np.testing.assert_array_equal(changed[:,:,8:],x[:,:,8:])
        assert np.all(changed[:,:,7]>=changed[:,:,6]) and np.all(changed[:,:,:8]>=0)
    else:
        for i in range(2):np.testing.assert_allclose(changed[i,-13:],point_features(changed[i,:,0])[-13:],atol=1e-6)


def test_coherent_cold_start_updates_repeated_first_observation():
    pytest.importorskip('pywt')
    from finad_repair.sdfmm import coherent,point_features
    x=np.repeat(point_features(np.array([.5]))[None],160,axis=1)
    changed=coherent(x,np.array([0]),'sd_stock',np.random.default_rng(2))
    np.testing.assert_array_equal(changed[0],np.repeat(changed[0,-1:],160,axis=0))


def test_sd_repair_reload_without_any_labels(tmp_path):
    pytest.importorskip('pywt')
    from finad_repair.sdfmm import SDRepair,point_features
    from finad_recent.sdfmm import sequence_starts
    raw=point_features(np.random.default_rng(3).uniform(.1,1,32));pre=Preprocessor().fit(raw)
    starts=sequence_starts(np.zeros(32,int));m=SDRepair(11,'sd_stock',steps=2,endpoint=True,generator=True).fit(raw,starts,np.arange(32),pre)
    m.save(tmp_path/'model.pt');loaded=SDRepair.load(tmp_path/'model.pt')
    np.testing.assert_allclose(m.score_raw(raw,starts,np.arange(32),pre),loaded.score_raw(raw,starts,np.arange(32),pre),atol=1e-6)


def test_planned_ablation_matrix_is_complete():
    pytest.importorskip('pywt')
    from finad_repair.experiment import planned
    jobs=planned();assert len(jobs)==174 and len(set(jobs))==174
    assert {f:sum(x[0]==f for x in jobs) for f in ['huge','himvh','sdfmm']}==dict(huge=24,himvh=42,sdfmm=108)


@pytest.mark.parametrize('family', ['huge','himvh','sdfmm'])
def test_repaired_pipeline_without_training_or_calibration_targets(family,tmp_path,monkeypatch):
    pytest.importorskip('pywt')
    import json
    from types import SimpleNamespace
    import finad_repair.experiment as run
    data=tmp_path/'tabular';graphs=tmp_path/'graphs';out=tmp_path/'out';out.mkdir()
    args=SimpleNamespace(data=data,graphs=graphs,device='cpu')
    rng=np.random.default_rng(6)
    if family=='huge':
        root=graphs/'banksim';root.mkdir(parents=True)
        for phase in ['train','calibration','test']:
            x=rng.uniform(0,2,(4,12,10)).astype(np.float32);x[:,:,0]=np.r_[np.ones(8),np.zeros(4)];x[:,:,1]=1-x[:,:,0]
            e=np.array([np.arange(8),np.tile(np.arange(8,12),2)])
            np.savez(root/f'{phase}.graph.npz',x=x,active=np.ones((4,12),bool),**{f'e{i}':e for i in range(4)})
        np.save(root/'test.y.npy',np.tile([0,1],6).astype(np.int8));(root/'manifest.json').write_text('{}')
        base=run.HUGE
        def factory(width,seed,device,steps):return base(width,seed,device,2)
        factory.load=base.load;monkeypatch.setattr(run,'HUGE',factory)
        result,_,_=run.graph_run(args,'banksim','role',42,out,{})
        assert result['evaluated']==8
    else:
        task='banksim' if family=='himvh' else 'sd_stock';p=61 if family=='himvh' else 11
        root=data/task;root.mkdir(parents=True)
        for phase in ['train','calibration','test']:
            x=rng.uniform(.1,1,(32,p)).astype(np.float32);x.tofile(root/f'{phase}.x.f32');np.zeros(32,np.int32).tofile(root/f'{phase}.seq.i32')
        np.tile([0,1],16).astype(np.int8).tofile(root/'test.y.i8')
        (root/'manifest.json').write_text(json.dumps(dict(features=[str(i) for i in range(p)],splits={s:{'rows':32} for s in ['train','calibration','test']})))
        if family=='himvh':
            monkeypatch.setattr(run,'transaction_keys',lambda task,phase,indices=None:([hyperedges(np.arange(32)//8)]*4,32))
            base=run.HIMVHRepair
            def factory(*a,**kw):kw['steps']=2;return base(*a,**kw)
            factory.load=base.load;monkeypatch.setattr(run,'HIMVHRepair',factory)
        else:
            base=run.SDRepair
            def factory(*a,**kw):kw['steps']=2;return base(*a,**kw)
            factory.load=base.load;monkeypatch.setattr(run,'SDRepair',factory)
        result,_,_=run.tabular_run(args,task,family,'cm' if family=='himvh' else 'ge',42,out,{})
        assert result['evaluated']==32
    assert not list(root.glob('train.y.*')) and not list(root.glob('calibration.y.*'))
