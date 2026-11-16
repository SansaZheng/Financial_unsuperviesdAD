"""Numerical author-formula checks and strict zero-real-label execution gates."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from finad_recent.huge import HUGE, HugeNet, halo, rank_loss, alignment, similarity, loop_edges
from finad_recent.himvh import HIMVHZero, NoveltyLayer, hyperedges, discrepancies


def test_huge_halo_matches_author_weighted_definition():
    x = torch.tensor([[1., 2.], [4., -1.], [0., 0.], [1., 2.]])
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    result = halo(x, edges).numpy()
    a, b = x.numpy()[edges[0]], x.numpy()[edges[1]]
    coefficient = np.abs(a-b)+1e-9
    a, b = a*coefficient, b*coefficient
    h = np.sqrt(np.sum((a-b)**2, axis=1))/np.sqrt(np.sum(a*a+b*b, axis=1)+1e-6)
    expected = np.asarray([h[1], (h[0]+h[3])/2, h[2], -1])
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_huge_gcn_matches_symmetric_dense_reference_and_gradients():
    torch.manual_seed(7)
    net = HugeNet(3, 5).double()
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    e = loop_edges(torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), 4)
    mlp, actual = net(x, e)
    adj = torch.zeros(4, 4, dtype=torch.float64)
    adj[e[1], e[0]] = 1
    d = adj.sum(1).rsqrt()
    expected = net.activation((adj*d[:, None]*d[None, :])@mlp@net.gcn_weight+net.gcn_bias)
    torch.testing.assert_close(actual, expected)
    g1 = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(g1, g2)


def test_huge_losses_and_negative_score_match_author_reference():
    p = torch.tensor([-.4, .2, .7], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([.3, .8, .1], dtype=torch.float64)
    s = p[None, :]-p[:, None]
    target = (h[None, :]-h[:, None] > 0).double()
    expected = ((1-target)*s+torch.log1p(torch.exp(-s))).mean()
    torch.testing.assert_close(rank_loss(p, h), expected)
    z = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], requires_grad=True)
    e = loop_edges(torch.tensor([[0, 1], [1, 0]]), 3)
    pos, neg = similarity(z, e, True)
    assert float(pos[0]) == pytest.approx(.5)
    assert float(neg[0]) == pytest.approx(2**-.5)
    assert float(pos[2]) == pytest.approx(1.)
    gnn = (z.detach()+.3).requires_grad_()
    loss = alignment(z, gnn, e)
    grads = torch.autograd.grad(loss, (z, gnn), allow_unused=True)
    assert grads[0] is not None and grads[1] is None  # asymmetric teacher stop-grad


def test_hyperedges_no_cross_key_and_literal_reverse_kl():
    keys = ['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b']
    edges = hyperedges(keys, 4)
    assert edges.tolist() == [[0, 2, 3, 4], [1, 5, 6, 7]]
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    features = discrepancies(x, [edges, edges])
    np.testing.assert_array_equal(features[:, -3:], np.zeros((8, 3)))
    assert len(hyperedges(['a', 'a'], 4)) == 0


def test_novelty_recipient_excludes_own_message():
    layer = NoveltyLayer(1).double()
    with torch.no_grad():
        layer.weight.weight.fill_(1)
    h = torch.tensor([[1.], [3.]], dtype=torch.float64)
    actual = layer(h, torch.tensor([[0, 1]]))
    # Both deviations have equal novelty; alpha=.5. Recipient 0 gets .5*3.
    torch.testing.assert_close(actual, torch.sigmoid(torch.tensor([[2.5], [3.5]], dtype=torch.float64)))


def test_causal_context_never_crosses_sequence_or_uses_future():
    pytest.importorskip('pywt')
    from finad_recent.sdfmm import sequence_starts, contexts
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    starts = sequence_starts(np.repeat([3, 4, 7], 4))
    before = contexts(x, starts, [5])
    assert before.shape == (1, 160, 2)
    np.testing.assert_array_equal(before[0, 0], x[4])
    np.testing.assert_array_equal(before[0, -1], x[5])
    x[6:] += 1000
    np.testing.assert_array_equal(before, contexts(x, starts, [5]))


def test_synthetic_targets_are_exact_injected_support():
    pytest.importorskip('pywt')
    from finad_recent.sdfmm import synthesize
    x = np.random.default_rng(4).normal(size=(4, 160, 3)).astype(np.float32)
    altered, mask, original, changed = synthesize(x, np.random.default_rng(8))
    np.testing.assert_array_equal(altered[mask == 0], x[mask == 0])
    assert np.all(mask.sum(1) == 13)
    assert np.all(mask[:, -13:] == 1) and np.all(mask[:, :-13] == 0)
    assert np.max(np.abs(changed-original)) > .1
    assert np.isfinite(altered).all()


@pytest.mark.parametrize('method', ['huge', 'himvh', 'sdfmm'])
def test_label_free_fit_reload_and_finite_scores(method, tmp_path):
    torch.set_num_threads(2)
    x = np.random.default_rng(1).normal(size=(12, 3)).astype(np.float32)
    e = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    if method == 'huge':
        model = HUGE(3, steps=2).fit(x, e); inputs = (x, e)
    elif method == 'himvh':
        views = [hyperedges(np.repeat([0, 1, 2], 4)), hyperedges(np.tile([0, 1, 2], 4))]
        model = HIMVHZero(3, 2, steps=2, hidden=8, layers=1).fit(x, views); inputs = (x, views)
    else:
        pytest.importorskip('pywt')
        from finad_recent.sdfmm import SDFMMZero, sequence_starts
        starts = sequence_starts(np.repeat([0, 1, 2], 4))
        model = SDFMMZero(3, steps=2).fit(x, starts, np.arange(12)); inputs = (x, starts, np.arange(12))
    model.save(tmp_path/'model.pt')
    loaded = type(model).load(tmp_path/'model.pt')
    scores = model.score(*inputs)
    assert len(scores) == len(x) and np.isfinite(scores).all()
    np.testing.assert_allclose(scores, loaded.score(*inputs), rtol=1e-5, atol=1e-6)


def test_target_access_after_checkpoint_scores_and_thresholds(tmp_path):
    pytest.importorskip('pywt')
    from finad_recent.experiment import finish
    (tmp_path/'model.pt').write_bytes(b'checkpoint')
    def read_test():
        for f in ['scores.npz', 'thresholds.json', 'label_access_gate.json']:
            assert (tmp_path/f).exists()
        return np.array([0, 1, -1])
    result = finish(tmp_path, np.linspace(0, 1, 200), np.array([.1, .9, .5]), read_test)
    assert result['unknown_excluded'] == 1


@pytest.mark.parametrize('method', ['himvh_zero_reconstruction', 'sdfmm_zero_synthetic'])
def test_full_tabular_runner_with_no_training_or_calibration_label_files(method, tmp_path, monkeypatch):
    pytest.importorskip('pywt')
    import json
    import finad_recent.experiment as runner
    data = tmp_path/'data'; out = tmp_path/'output'; data.mkdir(); out.mkdir()
    rng = np.random.default_rng(8)
    for phase in ['train', 'calibration', 'test']:
        rng.normal(size=(12, 3)).astype(np.float32).tofile(data/f'{phase}.x.f32')
        np.repeat(np.arange(3), 4).astype(np.int32).tofile(data/f'{phase}.seq.i32')
    np.tile([0, 1], 6).astype(np.int8).tofile(data/'test.y.i8')
    (data/'manifest.json').write_text(json.dumps(dict(features=['a', 'b', 'c'], splits={s: {'rows': 12} for s in ['train', 'calibration', 'test']})))
    views = [hyperedges(np.repeat(np.arange(3), 4)) for _ in range(4)]
    monkeypatch.setattr(runner, 'transaction_keys', lambda *a: (views, 12))
    args = SimpleNamespace(himvh_steps=2, sd_steps=2)
    result, _, _ = runner.tabular_run(data, out, 's_ffsd', method, 42, 'cpu', args)
    assert result['evaluated'] == 12
    assert not (data/'train.y.i8').exists() and not (data/'calibration.y.i8').exists()


def test_full_graph_runner_with_no_training_or_calibration_label_files(tmp_path):
    pytest.importorskip('pywt')
    import json
    from finad_recent.experiment import graph_run
    data = tmp_path/'graphs'; out = tmp_path/'output'; data.mkdir(); out.mkdir()
    rng = np.random.default_rng(9)
    for phase in ['train', 'calibration', 'test']:
        x = rng.normal(size=(4, 12, 10)).astype(np.float32)
        x[:, :, 0] = np.r_[np.ones(8), np.zeros(4)]
        x[:, :, 1] = 1-x[:, :, 0]
        e = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 8, 9, 10, 11]])
        np.savez(data/f'{phase}.graph.npz', x=x, active=np.ones((4, 12), bool), **{f'e{i}': e for i in range(4)})
    np.save(data/'test.y.npy', np.tile([0, 1], 6).astype(np.int8))
    (data/'manifest.json').write_text(json.dumps({'fixture': True}))
    result, _, _ = graph_run(data, out, 42, 'cpu', SimpleNamespace(huge_steps=2))
    assert result['evaluated'] == 8
    assert not (data/'train.y.npy').exists() and not (data/'calibration.y.npy').exists()
