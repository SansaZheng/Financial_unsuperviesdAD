import numpy as np
import pytest
import torch

from finad.classical import ClassicalDetector, METHODS as CLASSICAL
from finad.neural import NeuralDetector, VariationalAutoencoder, METHODS as NEURAL
from finad.preprocessing import Preprocessor

torch.set_num_threads(2)


@pytest.mark.parametrize('method', CLASSICAL+NEURAL)
def test_fit_score_batch_invariance_and_checkpoint(method, tmp_path):
    rng = np.random.default_rng(42)
    train = rng.normal(size=(96, 6)).astype(np.float32)
    query = rng.normal(size=(23, 6)).astype(np.float32)
    detector = (ClassicalDetector(method) if method in CLASSICAL else
                NeuralDetector(method, steps=4, pretrain_steps=3)).fit(train)
    score = detector.score(query)
    batched = np.concatenate([detector.score(query[:9]), detector.score(query[9:])])
    assert score.shape == (23,) and np.isfinite(score).all()
    np.testing.assert_allclose(score, batched, rtol=2e-5, atol=1e-5)
    path = tmp_path/('model.joblib' if method in CLASSICAL else 'model.pt')
    detector.save(path)
    restored = (ClassicalDetector.load(path) if method in CLASSICAL else NeuralDetector.load(path))
    np.testing.assert_allclose(score, restored.score(query), rtol=2e-5, atol=1e-5)


def test_knn_is_kth_distance_and_high_means_abnormal():
    x = np.arange(30, dtype=np.float32).reshape(-1, 1)
    model = ClassicalDetector('knn').fit(x)
    q = np.array([[10.5], [100]], dtype=np.float32)
    expected = np.sort(np.abs(q-x.T), axis=1)[:, 19]
    np.testing.assert_allclose(model.score(q), expected)
    assert model.score(q)[1] > model.score(q)[0]


def test_pca_is_squared_residual_not_density():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    model = ClassicalDetector('pca').fit(x)
    c = x-model.center
    residual = c-(c@model.components.T)@model.components
    np.testing.assert_allclose(model.score(x), np.sum(residual.astype(np.float64)**2, axis=1), atol=1e-5)


def test_hbos_training_only_histograms_and_constant_column():
    x = np.array([[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]], dtype=np.float32)
    model = ClassicalDetector('hbos').fit(x)
    before = [(a.copy(), b.copy()) for a, b, _ in model.histograms]
    assert model.score(np.array([[100., 0.]], dtype=np.float32))[0] > model.score(x).min()
    for (a, b, _), (old_a, old_b) in zip(model.histograms, before):
        np.testing.assert_array_equal(a, old_a); np.testing.assert_array_equal(b, old_b)


def test_svdd_bias_free_fixed_center_and_exact_score():
    x = np.random.default_rng(3).normal(size=(32, 5)).astype(np.float32)
    model = NeuralDetector('deep_svdd', steps=3, pretrain_steps=2).fit(x)
    assert all(layer.bias is None for layer in model.model.modules() if isinstance(layer, torch.nn.Linear))
    center = model.center.clone()
    with torch.no_grad():
        expected = (model.model(torch.as_tensor(x))-model.center).square().sum(1).numpy()
    np.testing.assert_allclose(model.score(x), expected)
    assert torch.equal(center, model.center) and (model.center.abs() >= .1).all()


def test_gaussian_vae_kl_and_deterministic_query_order():
    assert torch.equal(VariationalAutoencoder.kl(torch.zeros(4, 3), torch.zeros(4, 3)), torch.zeros(4))
    x = np.random.default_rng(2).normal(size=(32, 5)).astype(np.float32)
    model = NeuralDetector('vae', steps=3).fit(x)
    np.testing.assert_allclose(model.score(x), model.score(x[::-1].copy())[::-1], rtol=1e-5, atol=1e-5)


def test_train_only_imputation_clipping_and_reload(tmp_path):
    x = np.array([[1, 2, np.nan], [2, np.nan, np.nan], [3, 2, np.nan]])
    prep = Preprocessor().fit(x); center = prep.center.copy()
    query = np.array([[10000000, np.nan, 1000]])
    z = prep.transform(query)
    assert z.shape == (1, 3) and z[0, 0] == 8 and z[0, -1] == 1
    np.testing.assert_array_equal(prep.center, center)
    prep.save(tmp_path/'prep.npz')
    np.testing.assert_array_equal(z, Preprocessor.load(tmp_path/'prep.npz').transform(query))


def test_gmm_sparse_near_constant_features_have_finite_scores():
    rng=np.random.default_rng(7)
    x=np.full((1024,8),7.999,dtype=np.float32)
    x[:,0]=rng.normal(size=len(x));x[:,1]=rng.choice([0.,1.],len(x),p=[.999,.001])
    x[:,2]+=rng.normal(0,1e-6,len(x)).astype(np.float32)
    model=ClassicalDetector('gmm').fit(x)
    assert np.isfinite(model.score(x)).all()
    assert (model.model.covariances_>0).all()
