import numpy as np
import torch
from finad_kronos.adapter import history, sequence_starts, forecast, residual_scores, batch_seed


class FakePredictor:
    device = 'cpu'

    def generate(self, x, xs, ys, *args):
        output = np.zeros((len(x), 1, 6), dtype=np.float32)
        output[:, 0, 3] = x[:, -1, 3]+torch.rand(len(x)).numpy()*.01
        return output


def test_histories_are_strictly_past_and_reset():
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    starts = sequence_starts(np.array([0, 0, 0, 1, 1, 1]))
    h, n = history(values, np.array([0, 2, 3, 5]), np.array([0, 0, 1, 1]), starts)
    assert list(n) == [0, 2, 0, 2]
    np.testing.assert_array_equal(h[1, -2:], [0, 2])
    np.testing.assert_array_equal(h[3, -2:], [7, 9])
    assert np.all(h[[0, 2]] == 0)
    changed = values.copy(); changed[2:] = -999
    np.testing.assert_array_equal(history(values, [2], [0], starts)[0], history(changed, [2], [0], starts)[0])


def test_frozen_forecast_repeat_and_rng_isolation():
    h = np.arange(64).reshape(2, 32).astype(float)
    torch.manual_seed(999); before = torch.get_rng_state().clone()
    p, std = forecast(FakePredictor(), h, [32, 64], batch_seed('x', 42, 'calibration', 0))
    torch.testing.assert_close(before, torch.get_rng_state())
    again, _ = forecast(FakePredictor(), h, [32, 64], batch_seed('x', 42, 'calibration', 0))
    np.testing.assert_array_equal(p, again)
    np.testing.assert_allclose(std, h.std(1))
    assert p.dtype == np.float32


def test_residual_formula_cold_start_and_all_features_max():
    scores = residual_scores(np.array([[2., 3.], [0, 1]]), np.array([[1., 1.], [0., 0.]]), np.array([[1., 2.], [0., 0.]]))
    np.testing.assert_allclose(scores[0], [(1/(1+1e-6))**2, (2/(2+1e-6))**2])
    np.testing.assert_array_equal(scores.max(1), [scores[0].max(), 1e12])


def test_no_labels_in_preprocessing_or_forecast_contract():
    import inspect
    for function in [history, forecast, sequence_starts, residual_scores]:
        assert not any('label' in name or 'target_y' in name for name in inspect.signature(function).parameters)


def test_sequence_reentry_is_rejected():
    import pytest
    with pytest.raises(ValueError, match='contiguous'):
        sequence_starts([0, 1, 0])


def test_saved_forecasts_reconstruct_scores_and_resume(tmp_path):
    from finad_kronos.experiment import score_phase
    values = np.arange(120, dtype=np.float32).reshape(40, 3)/7
    starts = sequence_starts(np.repeat([0, 1], 20))
    targets = np.array([0, 1, 19, 20, 21, 39])
    predicted, score = score_phase(FakePredictor(), values, starts, targets, tmp_path, 'test', 'toy', 42, 8)
    expected = []
    for row, t in enumerate(targets):
        h, _ = history(values, np.repeat(t, 3), np.arange(3), starts)
        expected.append(residual_scores(values[t], predicted[row], h.std(1)).max())
    np.testing.assert_array_equal(score, expected)
    p2, s2 = score_phase(FakePredictor(), values, starts, targets, tmp_path, 'test', 'toy', 42, 8)
    np.testing.assert_array_equal(predicted, p2); np.testing.assert_array_equal(score, s2)


def test_full_run_opens_test_labels_only_after_frozen_artifacts(tmp_path, monkeypatch):
    import json
    from finad_kronos.experiment import run_one
    data, output = tmp_path/'data/toy', tmp_path/'results'
    data.mkdir(parents=True)
    sizes = dict(train=20, calibration=24, test=10)
    rng = np.random.default_rng(88)
    for split, n in sizes.items():rng.normal(size=(n, 3)).astype('float32').tofile(data/f'{split}.x.f32')
    (data/'manifest.json').write_text(json.dumps(dict(dataset='toy', features=['a', 'b', 'c'], splits={s:dict(rows=n) for s,n in sizes.items()})))
    np.array([0, 1, -1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8).tofile(data/'test.y.i8')
    original = np.fromfile; opened = []
    def guarded(path, *args, **kwargs):
        if str(path).endswith('.y.i8'):
            assert str(path).endswith('test.y.i8')
            for name in ['scores.npz', 'forecasts.npz', 'thresholds.json', 'model_spec.json']:
                assert (output/'toy/42'/name).exists()
            opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'fromfile', guarded)
    run_one(tmp_path/'data', output, FakePredictor(), 'toy', 42, 16)
    record = json.loads((output/'toy/42/result.json').read_text())
    assert len(opened)==1 and record['metrics']['evaluated']==9
    assert record['metrics']['unknown_excluded']==1 and record['checks']['full_test_scored']
