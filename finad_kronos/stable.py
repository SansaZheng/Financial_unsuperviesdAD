"""Label-free, fixed residual/cold-start rules and an explicit POT guard.

These rules were fixed for retrospective development on the existing test split.
They are not a distribution-free guarantee or a new backbone model.
"""
import numpy as np
from finad.thresholds import POT, calibrate

EPS = 1e-6
FLOOR_FRACTION = .1
LOOKBACK = 32
VARIANTS = ('legacy', 'n', 'c', 'nc', 'prior', 'persist')
METHODS = dict(legacy='kronos_potratio_matched', n='kronos_stable_n',
               c='kronos_stable_c', nc='kronos_stable_nc',
               prior='train_prior_potratio', persist='persistence_stable_nc')


def fit_prior(train):
    """Only the paired, preprocessed training sample is accepted."""
    x = np.asarray(train, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need a finite two-dimensional training sample')
    q1, q3 = np.quantile(x, [.25, .75], axis=0)
    robust, std = (q3-q1)/1.349, x.std(axis=0)
    scale = np.where(robust > 1e-8, robust, np.where(std > 1e-8, std, 1.))
    return dict(center=np.median(x, axis=0).tolist(), scale=scale.tolist(),
                floor_fraction=FLOOR_FRACTION, epsilon=EPS, lookback=LOOKBACK,
                training_rows=len(x), rule='training IQR/1.349; std fallback; then 1')


def mix_cold(model_error, prior_error, counts):
    counts = np.asarray(counts)
    if counts.shape != (len(model_error),) or np.any((counts < 0) | (counts > LOOKBACK)):
        raise ValueError('Invalid causal history counts')
    weight = counts[:, None]/LOOKBACK
    return weight*model_error+(1-weight)*prior_error


def errors(observed, predicted, history_std, counts, last, prior, variant):
    """Current observations enter residuals, never their own past statistics."""
    if variant not in VARIANTS:raise ValueError('Unknown fixed variant')
    z, p, std = (np.asarray(a, dtype=np.float64) for a in (observed, predicted, history_std))
    if z.ndim != 2 or p.shape != z.shape or std.shape != z.shape or np.any(std < 0):
        raise ValueError('Invalid component arrays')
    center, reference = np.asarray(prior['center']), np.asarray(prior['scale'])
    if center.shape != (z.shape[1],) or reference.shape != center.shape or np.any(reference <= 0):
        raise ValueError('Invalid training prior')
    prior_error = ((z-center)/(reference+EPS))**2
    if variant == 'prior':return prior_error
    denominator = np.maximum(std, FLOOR_FRACTION*reference)+EPS if variant in ('n','nc','persist') else std+EPS
    if variant == 'persist':
        predicted_last = np.asarray(last, dtype=np.float64)
        if predicted_last.shape != z.shape:raise ValueError('Invalid persistence observations')
        model_error = ((z-predicted_last)/denominator)**2
    else:model_error = ((z-p)/denominator)**2
    result = mix_cold(model_error, prior_error, counts) if variant in ('c','nc','persist') else model_error
    if not np.isfinite(result).all():raise ValueError('Nonfinite residual error')
    return result


def fit_feature_scales(calibration_errors, names):
    """Same per-dimension POT rule as the original matched scorer."""
    active, failed, tails = [], [], {}
    for j, name in enumerate(names):
        try:
            tail = POT.fit(calibration_errors[:, j], initial_quantile=.90, risk=.01, min_peaks=30)
            if tail.threshold <= 0:raise ValueError('Nonpositive residual threshold')
            active.append(j);tails[name]=tail.to_dict()
        except ValueError as exc:failed.append(dict(feature=name,index=j,reason=str(exc)))
    if not active:raise ValueError('All original per-feature POT fits failed; no fallback score is invented')
    return dict(active_indices=active,active_features=[names[j] for j in active],tails=tails,failed=failed,
                quantile=.90,risk=.01,min_peaks=30,
                source_rule='Only successfully fitted dimensions enter the max-ratio, matching the source method')


def aggregate(residuals, fitted):
    scales=np.asarray([fitted['tails'][name]['threshold'] for name in fitted['active_features']])
    result=(residuals[:,fitted['active_indices']]/scales).max(axis=1)
    if not np.isfinite(result).all():raise ValueError('Nonfinite scalar scores')
    return result


def guard_pot(calibration, common):
    """Only calibration scores; no test scores or targets in this interface."""
    cal=np.asarray(calibration,dtype=np.float64)
    if cal.ndim != 1 or not len(cal) or not np.isfinite(cal).all():raise ValueError('Invalid calibration scores')
    pot=common['pot'];reasons=[];raw_rate=None
    if pot['status']!='ok':reasons.append('original_fit_failed')
    else:
        raw_rate=float(np.mean(cal>=pot['threshold']))
        if pot['shape']>=1:reasons.append('shape_not_below_one')
        if not .005<=raw_rate<=.02:reasons.append('calibration_alarm_rate_outside_fixed_interval')
    use_pot=not reasons
    threshold=float(pot['threshold'] if use_pot else common['quantile']['threshold'])
    actual=float(np.mean(cal>=threshold))
    return dict(status='ok',threshold=threshold,policy='guarded_pot',
                selected='original_pot' if use_pot else 'quantile_fallback',
                rejection_reasons=reasons,original_pot_status=pot['status'],
                original_pot_shape=pot.get('shape'),original_pot_threshold=pot.get('threshold'),
                original_calibration_alarm_fraction=raw_rate,calibration_alarm_fraction=actual,
                accepted_calibration_alarm_interval=[.005,.02],shape_upper_exclusive=1.,
                fallback_quantile=.99,fallback_ties_outside_interval=not .005<=actual<=.02,
                finite_sample_coverage_guarantee=False)


def thresholds(calibration):
    common=calibrate(calibration)
    return {**common,'guarded_pot':guard_pot(calibration,common)}
