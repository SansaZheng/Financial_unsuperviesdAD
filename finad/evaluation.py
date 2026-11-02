"""Unadjusted binary metrics with explicit unknown-target masking."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc


def metrics(y, scores, thresholds):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.ndim != 1 or y.shape != scores.shape or not np.isin(y, [-1, 0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid target/score arrays')
    known = y != -1
    target, score = y[known], scores[known]
    result = {'evaluated': len(target), 'unknown_excluded': int((~known).sum()),
              'positives': int((target == 1).sum()), 'normal': int((target == 0).sum()),
              'prevalence': float(target.mean()) if len(target) else None}
    if len(np.unique(target)) == 2:
        p, r, _ = precision_recall_curve(target, score)
        result.update(average_precision=float(average_precision_score(target, score)),
                      roc_auc=float(roc_auc_score(target, score)), pr_auc_trapezoid=float(auc(r, p)))
    else:
        result.update(average_precision=None, roc_auc=None, pr_auc_trapezoid=None)
    result['operating_points'] = {}
    for name, threshold in thresholds.items():
        if threshold['status'] != 'ok' or not len(target):
            result['operating_points'][name] = None
            continue
        alarm = score >= threshold['threshold']
        tp = int(np.count_nonzero(alarm & (target == 1))); fp = int(np.count_nonzero(alarm & (target == 0)))
        fn = int(np.count_nonzero(~alarm & (target == 1))); tn = int(np.count_nonzero(~alarm & (target == 0)))
        precision, recall = tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.
        result['operating_points'][name] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
            precision=precision, recall=recall, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            fpr=fp/(fp+tn) if fp+tn else None, alarm_fraction=float(alarm.mean()))
    return result
