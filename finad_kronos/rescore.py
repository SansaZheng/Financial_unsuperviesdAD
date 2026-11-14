"""Faithful per-feature POT-ratio aggregation on the same frozen forecasts.

The primary method preserves the source detector's relative tail normalization.
The initial maximum-error scorer is retained as a declared ablation, not selected
or discarded by test performance. No additional model inference is performed.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
from finad.preprocessing import Preprocessor
from finad.thresholds import POT, calibrate
from finad.evaluation import metrics
from .adapter import sequence_starts
from .assets import sha
from .experiment import source_manifest, verify_inputs, write

METHOD = 'kronos_potratio_matched'


def fit_feature_tails(errors, names):
    active, failed, tails = [], [], {}
    for j, name in enumerate(names):
        try:
            tail = POT.fit(errors[:, j], initial_quantile=.90, risk=.01, min_peaks=30)
            if tail.threshold <= 0:
                raise ValueError('Nonpositive residual threshold')
            tails[name] = tail.to_dict(); active.append(j)
        except ValueError as error:
            failed.append(dict(feature=name, index=j, reason=str(error)))
    if not active:
        raise ValueError('All original per-feature POT fits failed; no fallback score is invented')
    return dict(active_indices=active, active_features=[names[j] for j in active], tails=tails, failed=failed,
                quantile=.90, risk=.01, min_peaks=30, source_rule='Only successfully fitted dimensions enter the max-ratio, matching the source method')


def aggregate_ratio(errors, fitted):
    indexes = fitted['active_indices']
    thresholds = np.array([fitted['tails'][name]['threshold'] for name in fitted['active_features']])
    score = (errors[:, indexes]/thresholds).max(axis=1)
    if not np.isfinite(score).all():raise ValueError('Nonfinite POT-normalized residual scores')
    return score


def residual_arrays(data, raw, destination):
    meta = json.loads((data/'manifest.json').read_text()); width = len(meta['features'])
    pre = Preprocessor.load(raw/'preprocessor.npz')
    xs = {s: np.memmap(data/f'{s}.x.f32', dtype='float32', mode='r', shape=(meta['splits'][s]['rows'], width)) for s in ('train', 'calibration', 'test')}
    values = np.concatenate([pre.transform(xs[s]) for s in ('train', 'calibration', 'test')])
    nt, nc = len(xs['train']), len(xs['calibration']); starts = np.zeros(len(values), dtype=np.int64)
    if meta.get('temporal'):
        starts = np.concatenate([sequence_starts(np.fromfile(data/f'{s}.seq.i32', dtype=np.int32))+offset for s, offset in [('train', 0), ('calibration', nt), ('test', nt+nc)]])
    with np.load(raw/'indices.npz', allow_pickle=False) as z:ci, ti = z['calibration'], z['test']
    arrays = {}
    with np.load(raw/'forecasts.npz', allow_pickle=False) as z:
        for phase, positions in [('calibration', nt+ci), ('test', nt+nc+ti)]:
            predicted = z[phase]
            assert predicted.shape == (len(positions), width)
            errors = np.empty(predicted.shape, dtype=np.float64)
            for begin in range(0, len(positions), 256):
                t = positions[begin:begin+256]; ix = t[:, None]-np.arange(32, 0, -1)
                valid = ix >= starts[t, None]; h = values[np.maximum(ix, 0)].astype(np.float64); h[~valid] = 0
                errors[begin:begin+len(t)] = ((values[t].astype(np.float64)-predicted[begin:begin+len(t)])/(h.std(axis=1)+1e-6))**2
            assert np.isfinite(errors).all(); arrays[phase] = errors
    for name in ['preprocessor.npz', 'indices.npz']:
        shutil.copyfile(raw/name, destination/name)
    link = destination/'forecasts.npz'
    if link.is_symlink():assert link.resolve() == (raw/'forecasts.npz').resolve()
    else:assert not link.exists();link.symlink_to((raw/'forecasts.npz').resolve())
    return arrays, meta


def main():
    p = argparse.ArgumentParser(); p.add_argument('--raw-runs', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    manifest = json.loads((a.raw_runs/'run_manifest.json').read_text())
    assert json.loads((a.raw_runs/'verification.json').read_text())['status'] == 'passed'
    a.output.mkdir(parents=True, exist_ok=True)
    scoring_manifest = dict(**source_manifest(), method=METHOD, tasks=manifest['tasks'], seeds=manifest['seeds'],
                            inference_manifest=manifest, inference_source_commit=manifest['git_commit'],
                            selection='Primary original per-feature POT-ratio aggregation specified before inspecting any full-cohort financial metrics; scalar max-error kept as ablation',
                            feature_pot=dict(initial_quantile=.90, risk=.01, min_peaks=30),
                            operating_points=['common scalar POT q=.85 risk=.01 min_peaks=20', 'common 99% calibration quantile'],
                            python=__import__('platform').python_version(),
                            packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','torch']})
    if (a.output/'run_manifest.json').exists() and json.loads((a.output/'run_manifest.json').read_text())!=scoring_manifest:
        raise ValueError('Scoring source/environment changed; use a new output directory')
    write(a.output/'run_manifest.json', scoring_manifest)
    write(a.output/'input_files_sha256.json', verify_inputs(a.data, manifest['tasks']))
    shutil.copyfile(a.raw_runs/'assets.json', a.output/'assets.json')
    records = []
    for d in manifest['tasks']:
        for seed in manifest['seeds']:
            started = time.monotonic(); raw = a.raw_runs/d/str(seed); out = a.output/d/str(seed); out.mkdir(parents=True, exist_ok=True)
            errors, meta = residual_arrays(a.data/d, raw, out)
            fitted = fit_feature_tails(errors['calibration'], meta['features'])
            scores = {phase: aggregate_ratio(errors[phase], fitted) for phase in ('calibration', 'test')}
            np.savez_compressed(out/'scores.npz', **scores); write(out/'feature_pot.json', fitted)
            thresholds = calibrate(scores['calibration']); write(out/'thresholds.json', thresholds)
            info = json.loads((raw/'model_spec.json').read_text())
            info.update(method=METHOD, scalar_score='max over valid dimensions of standardized squared residual / per-feature POT threshold',
                        feature_pot=dict(initial_quantile=.90, risk=.01, min_peaks=30), active_dimensions=len(fitted['active_indices']),
                        total_dimensions=len(meta['features']), failed_dimensions=fitted['failed'], original_max_error_variant='preserved as an ablation; not selected by metrics')
            write(out/'model_spec.json', info)
            # The source result and test labels are opened only after new scores,
            # fitted dimension scales and both fixed operating points are saved.
            prior = json.loads((raw/'result.json').read_text())
            y = np.fromfile(a.data/d/'test.y.i8', dtype=np.int8)
            record = dict(track='tabular', dataset=d, method=METHOD, seed=seed, seconds=prior['seconds']+time.monotonic()-started,
                          device=prior['device'], metrics=metrics(y, scores['test'], thresholds), model_info=info,
                          checks={**prior['checks'], 'raw_max_error_score_sha256': prior['checks']['score_sha256'],
                                  'score_sha256': sha(out/'scores.npz'), 'feature_pot_sha256': sha(out/'feature_pot.json'),
                                  'same_frozen_forecasts': True})
            write(out/'result.json', record); records.append(record)
            # Original OR operating point is descriptive only, not part of the
            # shared POT/quantile ranking or selected as a better threshold.
            write(out/'original_or_metrics.json', metrics(y, scores['test'], {'original_or': dict(status='ok', threshold=1.)}))
            print('PRESPECIFIED_POTRATIO_RESCORED', d, seed, flush=True)
    write(a.output/'records.json', records);write(a.output/'failures.json', [])
    write(a.output/'complete.json', dict(finished=len(records), failed=0, expected=len(manifest['tasks'])*len(manifest['seeds'])))


if __name__ == '__main__':main()
