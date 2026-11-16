import copy
import numpy as np
import pytest
from finad_kronos.stable import fit_prior,errors,mix_cold,guard_pot,fit_feature_scales,aggregate,thresholds


def prior():
    return dict(center=[0.,2.],scale=[1.,2.])


def test_training_reference_fallbacks_and_no_mutation():
    train=np.array([[0.,3.,0.],[0.,3.,0.],[0.,3.,0.],[1.,3.,4.]])
    original=train.copy();fitted=fit_prior(train)
    np.testing.assert_array_equal(train,original)
    np.testing.assert_allclose(fitted['center'],[0,3,0])
    assert fitted['scale'][1]==1 and np.all(np.asarray(fitted['scale'])>0)
    # Three quarters zero, one outlier => positive IQR in this sample.
    np.testing.assert_allclose(fitted['scale'][0],.25/1.349)
    heavy=np.r_[np.zeros(9),1.][:,None]
    np.testing.assert_allclose(fit_prior(heavy)['scale'],[heavy.std()])


def test_exact_legacy_and_floor_formula_at_zero_variance():
    z=np.array([[1.,4.],[2.,3.]]);p=np.zeros_like(z);std=np.array([[0.,0.],[.5,1.]])
    old=errors(z,p,std,np.array([0,32]),p,prior(),'legacy')
    stable=errors(z,p,std,np.array([0,32]),p,prior(),'n')
    np.testing.assert_allclose(old,(z/(std+1e-6))**2)
    np.testing.assert_allclose(stable,(z/(np.maximum(std,[.1,.2])+1e-6))**2)
    assert np.isfinite(stable).all() and stable[0,0]<101 and old[0,0]==1e12


def test_cold_prior_and_full_history_endpoints():
    z=np.array([[1.,4.],[2.,3.],[2.,3.]])
    p=np.zeros_like(z);std=np.ones_like(z);counts=np.array([0,16,32])
    base=errors(z,p,std,counts,p,prior(),'n')
    pe=errors(z,p,std,counts,p,prior(),'prior')
    combined=errors(z,p,std,counts,p,prior(),'nc')
    np.testing.assert_array_equal(combined[0],pe[0])
    np.testing.assert_array_equal(combined[2],base[2])
    np.testing.assert_array_equal(combined[1],.5*pe[1]+.5*base[1])
    assert len(combined)==len(z)  # Cold observations are scored, never dropped.
    with pytest.raises(ValueError):mix_cold(base,pe,np.array([0,33,32]))


@pytest.mark.parametrize('variant',['prior','persist'])
def test_backbone_free_controls_ignore_kronos_prediction(variant):
    z=np.array([[2.,4.],[3.,1.]]);p=z*.5;std=np.ones_like(z);counts=np.array([0,32]);last=np.zeros_like(z)
    a=errors(z,p,std,counts,last,prior(),variant)
    b=errors(z,p+1000,std,counts,last,prior(),variant)
    np.testing.assert_array_equal(a,b)


def test_guard_heavy_shape_explicitly_falls_back_without_modifying_common():
    cal=np.arange(1000.)
    common=dict(pot=dict(status='ok',shape=15.,threshold=1e19),quantile=dict(status='ok',threshold=float(np.quantile(cal,.99))))
    saved=copy.deepcopy(common);g=guard_pot(cal,common)
    assert common==saved and g['threshold']==common['quantile']['threshold']
    assert g['selected']=='quantile_fallback'
    assert set(g['rejection_reasons'])=={'shape_not_below_one','calibration_alarm_rate_outside_fixed_interval'}
    assert g['original_calibration_alarm_fraction']==0 and g['finite_sample_coverage_guarantee'] is False


def test_guard_accepts_fixed_interval_and_rejects_only_calibration_evidence():
    cal=np.arange(1000.)
    common=dict(pot=dict(status='ok',shape=.2,threshold=990.),quantile=dict(status='ok',threshold=989.01))
    g=guard_pot(cal,common)
    assert g['selected']=='original_pot' and g['calibration_alarm_fraction']==.01
    common['pot']['threshold']=800
    assert guard_pot(cal,common)['rejection_reasons']==['calibration_alarm_rate_outside_fixed_interval']


def test_failed_fit_and_discrete_fallback_do_not_invent_one_percent_alarm_rate():
    common=dict(pot=dict(status='failed',threshold=None,reason='degenerate'),quantile=dict(status='ok',threshold=1.))
    g=guard_pot(np.ones(100),common)
    assert g['selected']=='quantile_fallback' and g['calibration_alarm_fraction']==1.
    assert g['fallback_ties_outside_interval'] and g['original_pot_status']=='failed'
    assert g['rejection_reasons']==['original_fit_failed']


def test_guard_has_no_ranking_effect():
    cal=np.linspace(0,10,1000);before=cal.copy();result=thresholds(cal)
    np.testing.assert_array_equal(cal,before)
    assert set(result)=={'pot','quantile','guarded_pot'}
    assert result['quantile']['threshold']==np.quantile(cal,.99)


def test_scale_fit_records_constant_dimension_failure():
    rng=np.random.default_rng(92)
    err=np.column_stack([np.zeros(2000),rng.exponential(size=2000)])
    fit=fit_feature_scales(err,['constant','continuous'])
    assert fit['active_indices']==[1] and fit['failed'][0]['feature']=='constant'
    np.testing.assert_allclose(aggregate(err,fit),err[:,1]/fit['tails']['continuous']['threshold'])
    with pytest.raises(ValueError,match='All original'):fit_feature_scales(np.zeros((2000,2)),['a','b'])


def test_real_component_loader_resets_cases_excludes_future_and_never_decodes_labels(tmp_path,monkeypatch):
    import json
    from finad.preprocessing import Preprocessor
    from finad_kronos.stable_experiment import causal_components
    data=tmp_path/'data';raw=tmp_path/'raw';data.mkdir();raw.mkdir()
    xs=dict(train=np.arange(12,dtype=np.float32).reshape(6,2),
            calibration=np.arange(6,dtype=np.float32).reshape(3,2)+4,
            test=np.array([[1,2],[2,3],[3,4],[4,5]],dtype=np.float32))
    meta=dict(features=['a','b'],temporal=True,splits={k:dict(rows=len(v)) for k,v in xs.items()})
    (data/'manifest.json').write_text(json.dumps(meta))
    for k,v in xs.items():
        v.tofile(data/f'{k}.x.f32')
        seq=np.zeros(len(v),dtype=np.int32) if k!='test' else np.array([10,10,11,11],dtype=np.int32)
        seq.tofile(data/f'{k}.seq.i32')
    pre=Preprocessor().fit(xs['train']);pre.save(raw/'preprocessor.npz')
    np.savez(raw/'indices.npz',train=np.arange(6),calibration=np.arange(3),test=np.arange(4))
    np.savez(raw/'forecasts.npz',calibration=np.zeros((3,2),dtype=np.float32),test=np.zeros((4,2),dtype=np.float32))
    original=np.fromfile
    def no_labels(path,*args,**kwargs):
        assert not str(path).endswith('.y.i8')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'fromfile',no_labels)
    a,prior,_=causal_components(data,raw)
    np.testing.assert_array_equal(a['test']['counts'],[0,1,0,1])
    np.testing.assert_array_equal(a['test']['last'][[0,2]],np.tile(prior['center'],(2,1)))
    xs['test'][-1]+=1000;xs['test'].tofile(data/'test.x.f32')
    b,after,_=causal_components(data,raw)
    assert prior==after
    np.testing.assert_array_equal(a['test']['history_std'],b['test']['history_std'])
    np.testing.assert_array_equal(a['test']['last'],b['test']['last'])
