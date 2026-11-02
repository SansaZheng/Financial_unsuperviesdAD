"""Fixed upper-tail POT for real scores, including signed NLL/SVM scores."""
from dataclasses import asdict, dataclass
import warnings
import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True)
class POT:
    threshold: float
    initial_threshold: float
    shape: float
    scale: float
    n_calibration: int
    n_peaks: int
    initial_quantile: float
    risk: float

    @classmethod
    def fit(cls, scores, initial_quantile=.85, risk=.01, min_peaks=20):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError('POT requires finite 1D calibration scores')
        if not 0 < initial_quantile < 1 or not 0 < risk < 1-initial_quantile:
            raise ValueError('Invalid initial quantile/risk')
        u = float(np.quantile(scores, initial_quantile))
        peaks = scores[scores > u]-u
        if len(peaks) < min_peaks or np.ptp(peaks) == 0:
            raise ValueError(f'Degenerate POT tail: {len(peaks)} peaks, need >= {min_peaks} nondegenerate peaks')
        if risk >= len(peaks)/len(scores):
            raise ValueError('Risk must be smaller than the observed exceedance fraction')
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            try:
                shape, _, scale = genpareto.fit(peaks, floc=0)
            except (RuntimeWarning, ValueError, FloatingPointError) as error:
                raise ValueError('GPD fit failed') from error
        if not np.isfinite([shape, scale]).all() or scale <= 0:
            raise ValueError('Invalid GPD parameters')
        if not np.isfinite(genpareto.logpdf(peaks, shape, loc=0, scale=scale)).all():
            raise ValueError('GPD support excludes calibration peaks')
        threshold = u+float(genpareto.isf(risk*len(scores)/len(peaks), shape, loc=0, scale=scale))
        if not np.isfinite(threshold) or threshold <= u:
            raise ValueError('Invalid POT threshold')
        return cls(threshold, u, float(shape), float(scale), len(scores), len(peaks), initial_quantile, risk)

    def to_dict(self):
        return asdict(self)


def calibrate(scores, initial_quantile=.85, risk=.01):
    """Always disclose two prespecified thresholds; never silently replace POT."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Calibration scores must be a finite nonempty 1D array')
    result = {'quantile': {'status': 'ok', 'threshold': float(np.quantile(scores, 1-risk)),
                           'quantile': 1-risk}}
    try:
        result['pot'] = {'status': 'ok', **POT.fit(scores, initial_quantile, risk).to_dict()}
    except ValueError as error:
        result['pot'] = {'status': 'failed', 'reason': str(error), 'threshold': None}
    return result
