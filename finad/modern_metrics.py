"""Modern, sequence-aware evaluation of saved test scores. Labels are read only here.

Point view (pooled over the whole labelled test split, every task):
  AUC-ROC, AUC-PR (average precision), precision@n (n = number of positives),
  oracle best-F1 (label-dependent ceiling, never an operating point of the method),
  and at each fixed label-blind threshold: precision, recall, F1, FPR, MCC.

Sequence view (TSB-AD protocol, Liu & Paparrizos, NeurIPS 2024): every test sequence
that contains at least one anomaly is scored on its own and the per-sequence values
are macro-averaged, exactly as TSB-AD averages over its time series.
  threshold-free : AUC-PR, AUC-ROC, VUS-PR, VUS-ROC (Paparrizos et al., VLDB 2022),
                   PATE (Ghorbani et al., KDD 2024)
  fixed threshold: standard F1, PA-F1 (Xu et al., WWW 2018), PA%K-F1 area (Kim et al.,
                   AAAI 2022), event-based F1, range-based F1 (Tatbul et al., NeurIPS 2018),
                   affiliation precision/recall/F1 (Huet et al., KDD 2022), PATE-F1.
Pooled event view at a fixed threshold: share of all ground-truth events reached by an
alarm inside the event or within the tolerance, and the share of alarm episodes that lie
within the tolerance of some event.

The official TSB-AD (Apache-2.0) and PATE (MIT) packages compute their own metrics; MCC,
PA%K and the pooled event view are implemented here. Predictions use ``score >= threshold``,
the alarm rule of the benchmark.
"""
import math
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from .evaluation import metrics as base_metrics

PAK_GRID = tuple(range(0, 101, 10))
VUS_THRESHOLDS = 250


def segments(label):
    """Inclusive (start, end) pairs of the maximal runs of ones."""
    label = np.asarray(label).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], label, [0]])))
    return [(int(s), int(e)-1) for s, e in zip(edges[::2], edges[1::2])]


def mcc(tp, fp, fn, tn):
    denominator = math.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    return (float(tp)*tn-float(fp)*fn)/denominator if denominator else 0.


def point_metrics(y, scores, thresholds):
    """Pooled transaction/point metrics; extends the benchmark's unadjusted metrics."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    result = base_metrics(y, scores, thresholds)
    known = y != -1
    target, score = y[known], scores[known]
    if len(np.unique(target)) == 2:
        precision, recall, _ = precision_recall_curve(target, score)
        with np.errstate(invalid='ignore', divide='ignore'):
            f1 = np.where(precision+recall > 0, 2*precision*recall/(precision+recall), 0.)
        result['best_f1_oracle'] = float(np.max(f1))
        n = int((target == 1).sum())
        order = np.argsort(-score, kind='stable')
        result['precision_at_n'] = float(target[order[:n]].mean())
    else:
        result['best_f1_oracle'] = None; result['precision_at_n'] = None
    for name, op in result['operating_points'].items():
        if op is not None:
            op['mcc'] = mcc(op['tp'], op['fp'], op['fn'], op['tn'])
    return result


def pak_adjust(label, pred, k):
    """Point adjustment only inside ground-truth segments where more than k% of points are alarmed."""
    adjusted = pred.copy()
    for start, end in segments(label):
        if pred[start:end+1].mean()*100 > k:
            adjusted[start:end+1] = 1
    return adjusted


def pak_f1_area(label, pred, ks=PAK_GRID):
    return float(np.mean([f1_score(label, pak_adjust(label, pred, k), zero_division=0) for k in ks]))


def affiliation(label, pred):
    from TSB_AD.evaluation.affiliation.generics import convert_vector_to_events
    from TSB_AD.evaluation.affiliation.metrics import pr_from_events
    if pred.sum() == 0:
        return math.nan, 0., 0.
    events_pred, events_gt = convert_vector_to_events(pred), convert_vector_to_events(label)
    out = pr_from_events(events_pred, events_gt, (0, len(pred)))
    p, r = out['Affiliation_Precision'], out['Affiliation_Recall']
    f = 2*p*r/(p+r) if np.isfinite(p) and p+r > 0 else (0. if np.isfinite(p) else math.nan)
    return float(p), float(r), float(f)


def _sequence_blocks(seq):
    seq = np.asarray(seq)
    if seq.ndim != 1 or (seq < 0).any():
        raise ValueError('Sequence ids must be a 1D array of non-negative integers')
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(seq))+1, [len(seq)]])
    blocks = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
    ids = [int(seq[a]) for a, _ in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError('Rows of a sequence must be contiguous')
    return blocks


def _threshold_block(label, score, pred, tolerance):
    from TSB_AD.evaluation.basic_metrics import basic_metricor
    from pate.PATE_metric import PATE
    grader = basic_metricor()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p, r, f = affiliation(label, pred)
        return {'f1': float(f1_score(label, pred, zero_division=0)),
                'pa_f1': float(grader.metric_PointF1PA(label, score, preds=pred)),
                'pak_f1_area': pak_f1_area(label, pred),
                'event_f1': float(grader.metric_EventF1PA(label, score, preds=pred)),
                'range_f1': float(grader.metric_RF1(label, score, preds=pred)),
                'affiliation_precision': p, 'affiliation_recall': r, 'affiliation_f1': f,
                'pate_f1': float(PATE(label, pred, e_buffer=tolerance, d_buffer=tolerance, binary_scores=True)),
                'alarms': int(pred.sum())}


def pooled_event_view(y, scores, seq, threshold, tolerance):
    """Event recall (inside the event / within tolerance) and alarm-episode precision at one threshold."""
    alarm = (scores >= threshold).astype(np.int8)
    events = hit = hit_tol = episodes = episodes_ok = 0
    for a, b in _sequence_blocks(seq):
        label, pred = y[a:b], alarm[a:b]
        gt = segments(label)
        events += len(gt)
        for s, e in gt:
            hit += bool(pred[s:e+1].any())
            hit_tol += bool(pred[max(0, s-tolerance):e+tolerance+1].any())
        for s, e in segments(pred):
            episodes += 1
            episodes_ok += any(s <= ge+tolerance and e >= gs-tolerance for gs, ge in gt)
    recall, recall_tol = hit/events if events else None, hit_tol/events if events else None
    precision_tol = episodes_ok/episodes if episodes else None
    f1_tol = (2*precision_tol*recall_tol/(precision_tol+recall_tol) if precision_tol is not None and recall_tol is not None
              and precision_tol+recall_tol > 0 else (0. if events and episodes else None))
    return {'events': events, 'event_recall': recall, 'event_recall_tolerance': recall_tol,
            'alarm_episodes': episodes, 'alarm_episode_precision_tolerance': precision_tol,
            'event_f1_tolerance': f1_tol, 'alarm_fraction': float(alarm.mean())}


def sequence_metrics(y, scores, seq, thresholds, tolerance):
    """TSB-AD style per-sequence evaluation, macro-averaged over sequences with >=1 anomaly."""
    from TSB_AD.evaluation.basic_metrics import generate_curve
    from pate.PATE_metric import PATE
    y, scores = np.asarray(y).astype(np.int8), np.asarray(scores, dtype=np.float64)
    if (y == -1).any():
        raise ValueError('Sequence metrics require fully labelled sequences')
    tolerance = int(tolerance)
    if tolerance < 1:
        raise ValueError('Tolerance must be a positive number of rows')
    free, fixed = [], {name: [] for name, t in thresholds.items() if t['status'] == 'ok'}
    skipped = {'no_anomaly': 0, 'all_anomaly': 0}
    for a, b in _sequence_blocks(seq):
        label, score = y[a:b].astype(int), scores[a:b]
        if label.sum() == 0:
            skipped['no_anomaly'] += 1; continue
        if label.all():
            skipped['all_anomaly'] += 1; continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            *_, vus_roc, vus_pr = generate_curve(label, score, 2*tolerance, 'opt', VUS_THRESHOLDS)
            free.append({'auc_pr': float(average_precision_score(label, score)), 'auc_roc': float(roc_auc_score(label, score)),
                         'vus_pr': float(vus_pr), 'vus_roc': float(vus_roc),
                         'pate': float(PATE(label, score, e_buffer=tolerance, d_buffer=tolerance, n_jobs=1,
                                            num_desired_thresholds=VUS_THRESHOLDS)),
                         'length': int(b-a), 'anomalies': int(label.sum()), 'events': len(segments(label))})
        for name in fixed:
            pred = (score >= thresholds[name]['threshold']).astype(int)
            fixed[name].append(_threshold_block(label, score, pred, tolerance))

    def average(rows, key):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        return float(np.nanmean(values)) if np.isfinite(values).any() else None

    result = {'protocol': 'per-sequence macro average over test sequences with >=1 anomaly (TSB-AD)',
              'tolerance_rows': tolerance, 'vus_sliding_window': 2*tolerance, 'vus_thresholds': VUS_THRESHOLDS,
              'pak_grid': list(PAK_GRID), 'sequences_evaluated': len(free), 'sequences_skipped': skipped,
              'threshold_free': {k: average(free, k) for k in ('auc_pr', 'auc_roc', 'vus_pr', 'vus_roc', 'pate')} if free else None,
              'operating_points': {}}
    for name in thresholds:
        if name not in fixed or not fixed[name]:
            result['operating_points'][name] = None; continue
        rows = fixed[name]
        op = {k: average(rows, k) for k in ('f1', 'pa_f1', 'pak_f1_area', 'event_f1', 'range_f1',
                                             'affiliation_precision', 'affiliation_recall', 'affiliation_f1', 'pate_f1')}
        op['sequences_without_alarm'] = int(sum(r['alarms'] == 0 for r in rows))
        op['pooled_events'] = pooled_event_view(y, scores, seq, thresholds[name]['threshold'], tolerance)
        result['operating_points'][name] = op
    return result
