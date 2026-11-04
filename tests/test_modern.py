import json
import numpy as np
import pytest
import torch

from finad.rgld import RGLD, rank_reference
from finad.rttad import RTTAD, gamma_confidence


def test_average_tie_training_rank_and_inductive_extremes():
    ref=np.array([0.,1.,1.,3.])
    np.testing.assert_allclose(rank_reference(ref,np.array([-1.,0.,1.,3.,4.])),[0.,0.,.5,1.,1.])


def test_rgld_query_chunking_order_and_checkpoint(tmp_path):
    rng=np.random.default_rng(8);train=rng.normal(size=(80,6)).astype(np.float32)
    query=np.concatenate([train[:5],rng.normal(size=(25,6))]).astype(np.float32)
    model=RGLD(42,global_views=3,local_views=2,random_features=4,references=20).fit(train)
    original=model.score(query)
    np.testing.assert_allclose(original,np.concatenate([model.score(query[i:i+3]) for i in range(0,len(query),3)]))
    np.testing.assert_allclose(original,model.score(query[::-1])[::-1])
    assert np.isfinite(original).all() and ((original>=0)&(original<=1)).all()
    model.save(tmp_path/'model.joblib');np.testing.assert_allclose(original,RGLD.load(tmp_path/'model.joblib').score(query))


def test_rgld_constant_training_is_explicitly_finite():
    model=RGLD(global_views=2,local_views=2,random_features=2,references=20).fit(np.ones((30,3),np.float32))
    np.testing.assert_allclose(model.score(np.ones((4,3),np.float32)),.5)


def test_rttad_uses_both_adaptation_signs_and_persists_frozen_state(tmp_path,monkeypatch):
    import finad.rttad as module
    torch.set_num_threads(2);rng=np.random.default_rng(3)
    x=rng.normal(size=(64,5)).astype(np.float32)
    model=RTTAD(42,steps=3,adapt_rounds=1,adapt_steps=2).fit(x)
    old={k:v.clone() for k,v in model.model.state_dict().items()}
    monkeypatch.setattr(module,'gamma_confidence',lambda scores,seed:((np.arange(40),np.arange(40,64)),dict(status='test_fixture_selection')))
    model.adapt(x)
    assert model.info['adaptation'][0]['status']=='updated'
    assert model.info['adaptation'][0]['normal_selected']==40
    assert model.info['adaptation'][0]['anomaly_selected']==24
    assert any(not torch.equal(old[k],v) for k,v in model.model.state_dict().items())
    score=model.score(x);model.save(tmp_path/'model.pt');loaded=RTTAD.load(tmp_path/'model.pt')
    np.testing.assert_allclose(score,loaded.score(x),rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(score,np.concatenate([loaded.score(x[:16]),loaded.score(x[16:])]),rtol=1e-6,atol=1e-6)


def test_real_gamma_gmm_keeps_rng_and_reports_failure_instead_of_fallback():
    pytest.importorskip('pythresh')
    x=np.concatenate([np.linspace(.1,1.,100),np.linspace(4.,10.,20)])
    np.random.seed(19);before=np.random.get_state()
    selected,record=gamma_confidence(x,42)
    after=np.random.get_state();np.testing.assert_array_equal(before[1],after[1])
    assert record['status'] in ('ok','gamma_failure','no_positive_gamma_posterior')
    if selected is not None:assert not np.intersect1d(*selected).size
    json.dumps(record,allow_nan=False)


def test_modern_pipeline_without_train_or_calibration_labels(tmp_path):
    from argparse import Namespace
    from finad.experiment import benchmark
    from finad.verify import verify
    root=tmp_path/'cache'/'toy';root.mkdir(parents=True);rng=np.random.default_rng(5)
    manifest=dict(features=['a','b','c'],protocol='invented fixtures only',splits={})
    for split,n in [('train',80),('calibration',200),('test',40)]:
        rng.normal(size=(n,3)).astype(np.float32).tofile(root/f'{split}.x.f32');manifest['splits'][split]={'rows':n}
    np.array([0]*30+[1]*10,np.int8).tofile(root/'test.y.i8');(root/'manifest.json').write_text(json.dumps(manifest))
    args=Namespace(command='benchmark',cache=tmp_path/'cache',output=tmp_path/'runs',datasets=['toy'],methods=['rgld2026'],
        seeds=[42],train_cap=64,calibration_cap=200,steps=2,pretrain_steps=2,batch_size=32,risk=.01,initial_quantile=.85,device='cpu')
    benchmark(args);r=args.output/'toy'/'rgld2026_seed42'
    assert (r/'test_scores.npz').exists() and not (r/'test_scores.npy').exists()
    result=json.loads((r/'result.json').read_text());assert result['metrics']['evaluated']==40
    assert verify(args.output,args.cache)['verified_runs']==1
