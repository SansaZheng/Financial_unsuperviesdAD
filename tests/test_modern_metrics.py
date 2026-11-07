import numpy as np
import pytest

pytest.importorskip('TSB_AD')
pytest.importorskip('pate')
from finad.modern_metrics import (mcc, pak_adjust, pak_f1_area, point_metrics, pooled_event_view, segments,
                                  sequence_metrics)


def thresholds(value):
    return {'quantile': {'status': 'ok', 'threshold': value}, 'pot': {'status': 'failed', 'threshold': None}}


def test_segments_and_mcc():
    assert segments([0, 1, 1, 0, 1, 0, 0, 1]) == [(1, 2), (4, 4), (7, 7)]
    assert segments(np.zeros(5)) == []
    assert mcc(10, 0, 0, 10) == 1. and mcc(0, 10, 10, 0) == -1. and mcc(0, 0, 5, 5) == 0.


def test_point_metrics_extend_base_metrics():
    y = np.array([0, 0, 1, 0, 1, -1, 0, 0]); s = np.array([.1, .2, .9, .3, .8, 5., .05, .4])
    r = point_metrics(y, s, thresholds(.5))
    assert r['unknown_excluded'] == 1 and r['best_f1_oracle'] == 1. and r['precision_at_n'] == 1.
    op = r['operating_points']['quantile']
    assert op['f1'] == 1. and op['mcc'] == 1. and r['operating_points']['pot'] is None


def test_pak_adjustment_interpolates_between_pa_and_plain_f1():
    label = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0]); pred = np.array([0, 1, 0, 0, 0, 0, 0, 1, 1, 0])
    assert pak_adjust(label, pred, 0).tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]      # plain point adjustment
    assert pak_adjust(label, pred, 50).tolist() == [0, 1, 0, 0, 0, 0, 0, 1, 1, 0]     # 25% < 50%: no adjustment
    assert pak_adjust(label, pred, 100).tolist() == pred.tolist()
    assert 0 < pak_f1_area(label, pred) < 1


def test_sequence_metrics_match_official_single_sequence_values():
    from TSB_AD.evaluation.metrics import get_metrics
    rng = np.random.default_rng(1)
    y = np.zeros(400, int); y[100:110] = 1; y[300] = 1
    s = rng.random(400); s[95:112] += .7; s[298:302] += .5
    pred = (s >= 1.).astype(int)
    official = get_metrics(s, y, slidingWindow=8, pred=pred)
    mine = sequence_metrics(y, s, np.zeros(400, int), thresholds(1.), tolerance=4)
    free, op = mine['threshold_free'], mine['operating_points']['quantile']
    assert free['auc_pr'] == pytest.approx(official['AUC-PR']) and free['auc_roc'] == pytest.approx(official['AUC-ROC'])
    assert free['vus_pr'] == pytest.approx(official['VUS-PR']) and free['vus_roc'] == pytest.approx(official['VUS-ROC'])
    assert op['f1'] == pytest.approx(official['Standard-F1'], abs=1e-4) and op['pa_f1'] == pytest.approx(official['PA-F1'])
    assert op['event_f1'] == pytest.approx(official['Event-based-F1']) and op['range_f1'] == pytest.approx(official['R-based-F1'])
    assert op['affiliation_f1'] == pytest.approx(official['Affiliation-F'], abs=1e-6)
    assert 0 <= free['pate'] <= 1 and 0 <= op['pate_f1'] <= 1 and op['sequences_without_alarm'] == 0


def test_sequence_metrics_perfect_and_empty_predictions():
    y = np.array([0]*20+[1]*3+[0]*20+[0]*10+[1]*2+[0]*10); seq = np.array([0]*43+[1]*22)
    perfect = sequence_metrics(y, y.astype(float), seq, thresholds(.5), tolerance=2)
    assert perfect['sequences_evaluated'] == 2
    assert all(v == pytest.approx(1.) for v in perfect['threshold_free'].values())
    op = perfect['operating_points']['quantile']
    assert all(op[k] == pytest.approx(1.) for k in ('f1', 'pa_f1', 'pak_f1_area', 'event_f1', 'range_f1', 'affiliation_f1', 'pate_f1'))
    assert op['pooled_events'] == {'events': 2, 'event_recall': 1., 'event_recall_tolerance': 1., 'alarm_episodes': 2,
                                   'alarm_episode_precision_tolerance': 1., 'event_f1_tolerance': 1., 'alarm_fraction': 5/65}
    silent = sequence_metrics(y, np.zeros(65), seq, thresholds(.5), tolerance=2)['operating_points']['quantile']
    assert silent['sequences_without_alarm'] == 2 and silent['f1'] == 0 and silent['affiliation_f1'] == 0
    assert silent['affiliation_precision'] is None and silent['pooled_events']['event_recall'] == 0


def test_sequence_metrics_skip_sequences_without_anomaly_and_reject_split_sequences():
    y = np.array([0]*10+[1]+[0]*9+[0]*20); seq = np.array([0]*20+[1]*20)
    r = sequence_metrics(y, np.random.default_rng(0).random(40), seq, thresholds(.9), tolerance=2)
    assert r['sequences_evaluated'] == 1 and r['sequences_skipped'] == {'no_anomaly': 1, 'all_anomaly': 0}
    with pytest.raises(ValueError):
        sequence_metrics(y, np.zeros(40), np.array([0]*10+[1]*10+[0]*10+[1]*10), thresholds(.5), tolerance=2)


def test_pooled_event_view_uses_tolerance_for_near_misses():
    y = np.zeros(50, np.int8); y[20:23] = 1
    s = np.zeros(50); s[17] = 1.; s[40] = 1.   # one alarm 3 steps early, one far away
    view = pooled_event_view(y, s, np.zeros(50, int), .5, tolerance=3)
    assert view['event_recall'] == 0 and view['event_recall_tolerance'] == 1
    assert view['alarm_episodes'] == 2 and view['alarm_episode_precision_tolerance'] == .5
