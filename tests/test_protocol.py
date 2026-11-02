import json
from pathlib import Path
import numpy as np
import pytest

from finad.data import Cache, sample
from finad.evaluation import metrics
from finad.thresholds import POT, calibrate


def test_signed_pot_translation_invariance():
    x = np.random.default_rng(42).exponential(size=10000)
    a, b = POT.fit(x), POT.fit(x-100)
    assert b.threshold < 0
    assert abs((a.threshold-100)-b.threshold) < .001


def test_degenerate_pot_is_explicit_and_quantile_is_separate():
    result = calibrate(np.ones(1000))
    assert result['pot']['status'] == 'failed' and result['pot']['threshold'] is None
    assert result['quantile']['status'] == 'ok' and result['quantile']['threshold'] == 1


def test_unknown_labels_and_no_point_adjustment():
    t = {'pot': {'status':'ok','threshold':.5}}
    result = metrics(np.array([1,0,-1,1,0]), np.array([.9,.8,100,.1,.2]), t)
    assert result['unknown_excluded'] == 1 and result['evaluated'] == 4
    assert result['operating_points']['pot']['f1'] == .5
    assert {k:result['operating_points']['pot'][k] for k in ['tp','fp','fn','tn']} == dict(tp=1,fp=1,fn=1,tn=1)


def test_unlabelled_test_has_no_invented_metrics():
    result = metrics(np.array([-1,-1]), np.array([0.,1.]), {'pot':{'status':'ok','threshold':.5}})
    assert result['average_precision'] is None and result['operating_points']['pot'] is None


def test_uniform_sampling_is_reproducible_without_targets():
    x = np.arange(200).reshape(100, 2)
    a, i = sample(x, 13, 42); b, j = sample(x, 13, 42)
    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(i, j)
    assert len(np.unique(i)) == 13


def test_cache_can_fit_without_any_training_label_files(tmp_path):
    root = tmp_path/'toy'; root.mkdir()
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    x.tofile(root/'train.x.f32')
    (root/'manifest.json').write_text(json.dumps({'features':['a','b','c'],'splits':{'train':{'rows':8}}}))
    np.testing.assert_array_equal(Cache(tmp_path,'toy').x('train'), x)
    with pytest.raises(FileNotFoundError):
        Cache(tmp_path,'toy').y('train')
