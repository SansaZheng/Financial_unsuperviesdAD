"""Independent saved-forecast, paired-input and metric replay; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from finad.preprocessing import Preprocessor

p = argparse.ArgumentParser(); p.add_argument('--results', type=Path, required=True)
p.add_argument('--data', type=Path, required=True); args = p.parse_args()
records = json.loads((args.results/'records.json').read_text()); seen = set(); checks = []
golden = json.loads((Path(__file__).resolve().parents[1]/'results/zero_label_20261114/input_cache_files_sha256.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(2**20): h.update(b)
    return h.hexdigest()


for r in records:
    key = (r['dataset'], r['seed']); assert key not in seen; seen.add(key)
    data = args.data/r['dataset']; run = args.results/r['dataset']/str(r['seed'])
    for name, value in golden.items():
        if name.split('/')[0] == r['dataset']:
            path = args.data/name
            assert path.stat().st_size == value['bytes'] and sha(path) == value['sha256'], name
    meta = json.loads((data/'manifest.json').read_text()); width = len(meta['features'])
    x = {s: np.memmap(data/f'{s}.x.f32', mode='r', dtype='float32', shape=(meta['splits'][s]['rows'], width)) for s in ('train', 'calibration', 'test')}
    with np.load(run/'indices.npz', allow_pickle=False) as z:
        ti, ci, test_indices = z['train'], z['calibration'], z['test']
    expected_ti = np.sort(np.random.default_rng(r['seed']).choice(len(x['train']), min(8192, len(x['train'])), replace=False))
    expected_ci = np.sort(np.random.default_rng(r['seed']+100).choice(len(x['calibration']), min(100000, len(x['calibration'])), replace=False))
    np.testing.assert_array_equal(ti, expected_ti); np.testing.assert_array_equal(ci, expected_ci)
    np.testing.assert_array_equal(test_indices, np.arange(len(x['test'])))
    fitted, saved = Preprocessor().fit(x['train'][ti]), Preprocessor.load(run/'preprocessor.npz')
    for k in ('keep', 'missing', 'center', 'scale'):
        np.testing.assert_array_equal(getattr(fitted, k), getattr(saved, k))
    values = np.concatenate([saved.transform(x[s]) for s in ('train', 'calibration', 'test')])
    nt, nc = len(x['train']), len(x['calibration'])
    starts = np.zeros(len(values), dtype=np.int64)
    if meta.get('temporal'):
        for s, offset in [('train', 0), ('calibration', nt), ('test', nt+nc)]:
            seq = np.fromfile(data/f'{s}.seq.i32', dtype=np.int32)
            boundaries = np.r_[0, np.flatnonzero(np.diff(seq))+1, len(seq)]
            for a, b in zip(boundaries[:-1], boundaries[1:]): starts[offset+a:offset+b] = offset+a
    assert sha(run/'scores.npz') == r['checks']['score_sha256']
    assert sha(run/'forecasts.npz') == r['checks']['forecasts_sha256']
    with np.load(run/'scores.npz', allow_pickle=False) as z: cal, score = z['calibration'], z['test']
    with np.load(run/'forecasts.npz', allow_pickle=False) as z:
        for phase, positions, stored in [('calibration', nt+ci, cal), ('test', nt+nc+test_indices, score)]:
            predicted = z[phase]; assert predicted.shape == (len(positions), values.shape[1])
            for a in range(0, len(positions), 256):
                t = positions[a:a+256]; indexes = t[:, None]-np.arange(32, 0, -1)
                mask = indexes >= starts[t, None]
                h = values[np.maximum(indexes, 0)].astype(np.float64)
                h[~mask] = 0
                std = h.std(axis=1, ddof=0)
                residual = ((values[t].astype(np.float64)-predicted[a:a+len(t)])/(std+1e-6))**2
                np.testing.assert_allclose(residual.max(axis=1), stored[a:a+len(t)], rtol=1e-12, atol=1e-12)
    # Labels are opened only for this post-experiment evaluation replay.
    y = np.fromfile(data/'test.y.i8', dtype=np.int8)
    assert sha(data/'test.y.i8') == r['checks']['test_target_sha256'] and len(y) == len(score)
    known = y != -1; yy, ss = y[known], score[known]; m = r['metrics']
    assert m['unknown_excluded'] == int((~known).sum()) and m['positives'] == int(yy.sum()) and m['normal'] == int((yy==0).sum())
    precision, recall, _ = precision_recall_curve(yy, ss)
    for name, value in [('average_precision', average_precision_score(yy, ss)), ('roc_auc', roc_auc_score(yy, ss)), ('pr_auc_trapezoid', auc(recall, precision))]:
        assert np.isclose(value, m[name], rtol=1e-12, atol=1e-12), (key, name)
    thresholds = json.loads((run/'thresholds.json').read_text())
    for name, th in thresholds.items():
        if th['status'] != 'ok':
            assert m['operating_points'][name] is None
            continue
        if name == 'quantile':
            assert th['threshold'] == float(np.quantile(cal, .99))
        else:
            u = float(np.quantile(cal, .85)); peaks = cal[cal>u]-u
            assert len(peaks) >= 20 and np.ptp(peaks) > 0
            expected = u+float(genpareto.isf(.01*len(cal)/len(peaks), th['shape'], loc=0, scale=th['scale']))
            assert np.isclose(expected, th['threshold'], rtol=1e-12, atol=1e-12)
        alarm = ss >= th['threshold']; tp = int(np.sum((yy==1)&alarm)); fp = int(np.sum((yy==0)&alarm))
        fn = int(np.sum((yy==1)&~alarm)); tn = int(np.sum((yy==0)&~alarm)); op = m['operating_points'][name]
        assert [op[k] for k in ('tp', 'fp', 'fn', 'tn')] == [tp, fp, fn, tn]
        for metric, value in [('precision', tp/(tp+fp) if tp+fp else 0.), ('recall', tp/(tp+fn) if tp+fn else 0.), ('f1', 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.), ('fpr', fp/(fp+tn) if fp+tn else 0.)]:
            assert np.isclose(op[metric], value, rtol=1e-12, atol=1e-12), (key, metric)
    checks.append(dict(dataset=r['dataset'], seed=r['seed'], full_test_rows=len(y), per_feature_forecasts_replayed=len(values[0])*(len(cal)+len(score))))
    print('VERIFIED', *key, flush=True)
manifest = json.loads((args.results/'run_manifest.json').read_text())
assert seen == {(d, s) for d in manifest['tasks'] for s in manifest['seeds']}
assert json.loads((args.results/'complete.json').read_text()) == dict(finished=len(seen), failed=0, expected=len(seen))
(args.results/'verification.json').write_text(json.dumps(dict(status='passed', runs=len(checks), checks=['exact baseline cache hashes', 'paired train/calibration/test positions', 'train-only scaler reconstruction', 'all saved forecasts to scalar scores', 'AP/PR-AUC/ROC-AUC', 'POT and quantile', 'explicit confusion counts and P/R/F1/FPR'], verified=checks), indent=2)+'\n')
print('KRONOS_MATCHED_FULL_COHORT_VERIFIED', flush=True)
