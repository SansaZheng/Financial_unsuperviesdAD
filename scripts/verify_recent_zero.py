"""Independent saved-score replay and exact pairing with prior formal runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc
from finad.preprocessing import Preprocessor


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def equal(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def arrays_equal(left, right, keys=None):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for k in keys or a.files:
            np.testing.assert_array_equal(a[k], b[k])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--old-runs', type=Path, required=True); args = p.parse_args()
    complete = json.loads((args.runs/'complete.json').read_text())
    assert complete == dict(finished=30, expected=30, failed=0), complete
    records = json.loads((args.runs/'records.json').read_text()); assert len(records) == 30
    manifest = json.loads((args.runs/'run_manifest.json').read_text())
    assert manifest['seeds'] == [42, 43, 44]
    assert manifest['budgets'] == dict(huge=300, himvh=200, sdfmm=400)
    root = Path(__file__).resolve().parents[1]
    for path, expected in manifest['source_sha256'].items():
        content = subprocess.check_output(['git', '-C', str(root), 'show', manifest['git_commit']+':'+path])
        assert hashlib.sha256(content).hexdigest() == expected, path
    inputs = json.loads((args.runs/'input_files_sha256.json').read_text())
    for path, expected in inputs.items():
        kind, name = path.split('/', 1)
        f = (args.data if kind == 'tabular' else args.graphs)/name
        assert f.stat().st_size == expected['bytes'] and sha(f) == expected['sha256'], path
    seen = set(); verified = []
    for r in records:
        key = (r['track'], r['dataset'], r['method'], r['seed']); assert key not in seen; seen.add(key)
        track, task, method, seed = key
        run = args.runs/track/task/method/str(seed)
        prior = args.old_runs/track/task/'ae'/str(seed)
        assert sha(run/'scores.npz') == r['score_sha256']
        assert sha(run/'model.pt') == r['checkpoint_sha256']
        gate = json.loads((run/'label_access_gate.json').read_text())
        assert gate['status'] == 'scores_and_thresholds_saved_before_labels'
        for field, name in [('checkpoint_sha256', 'model.pt'), ('scores_sha256', 'scores.npz'), ('thresholds_sha256', 'thresholds.json')]:
            assert gate[field] == sha(run/name)
        with np.load(run/'scores.npz', allow_pickle=False) as z:
            cal, score = z['calibration'], z['test']
        with np.load(run/'indices.npz', allow_pickle=False) as z:
            indices = {k: z[k] for k in z.files}
        if track == 'tabular':
            source = args.data/task; meta = json.loads((source/'manifest.json').read_text())
            width = len(meta['features'])
            raw = np.memmap(source/'train.x.f32', mode='r', dtype='float32', shape=(meta['splits']['train']['rows'], width))
            ti = np.sort(np.random.default_rng(seed).choice(len(raw), min(8192, len(raw)), replace=False))
            ci = np.sort(np.random.default_rng(seed+100).choice(meta['splits']['calibration']['rows'], min(100000, meta['splits']['calibration']['rows']), replace=False))
            np.testing.assert_array_equal(indices['train'], ti); np.testing.assert_array_equal(indices['calibration'], ci)
            np.testing.assert_array_equal(indices['test'], np.arange(meta['splits']['test']['rows']))
            arrays_equal(run/'indices.npz', prior/'indices.npz', ['train', 'calibration'])
            fitted = Preprocessor().fit(raw[ti]); label_file = source/'test.y.i8'
            y = np.fromfile(label_file, dtype=np.int8)
        else:
            source = args.graphs/task
            with np.load(source/'train.graph.npz', allow_pickle=False) as z:
                fitted = Preprocessor().fit(z['x'].reshape(-1, 10))
            with np.load(source/'test.graph.npz', allow_pickle=False) as z:
                role = z['x'][0, :, 0] > .5
            np.testing.assert_array_equal(indices['test'], np.flatnonzero(role))
            label_file = source/'test.y.npy'; y = np.load(label_file, allow_pickle=False)[role]
            arrays_equal(run/'indices.npz', prior/'indices.npz')
        assert r['checks']['test_target_sha256'] == sha(label_file)
        old = json.loads((prior/'result.json').read_text())
        assert old['checks']['test_target_sha256'] == sha(label_file)
        arrays_equal(run/'preprocessor.npz', prior/'preprocessor.npz')
        saved = Preprocessor.load(run/'preprocessor.npz')
        for k in ['keep', 'missing', 'center', 'scale']:
            np.testing.assert_array_equal(getattr(saved, k), getattr(fitted, k))
        assert len(cal) == len(indices['calibration']) and len(y) == len(score)
        assert np.isfinite(cal).all() and np.isfinite(score).all()
        known = y != -1; s = score[known]; y = y[known]; m = r['metrics']
        assert m['unknown_excluded'] == int((~known).sum())
        assert m['evaluated'] == len(y) and m['positives'] == int(y.sum()) and m['normal'] == int((y == 0).sum())
        for k in ['evaluated', 'unknown_excluded', 'positives', 'normal', 'prevalence']:
            equal(m[k], old['metrics'][k])
        equal(m['prevalence'], np.mean(y))
        pr, re, _ = precision_recall_curve(y, s)
        for k, v in [('average_precision', average_precision_score(y, s)), ('pr_auc_trapezoid', auc(re, pr)), ('roc_auc', roc_auc_score(y, s))]:
            equal(m[k], v)
        thresholds = json.loads((run/'thresholds.json').read_text())
        u = np.quantile(cal, .85); peaks = cal[cal > u]-u
        can_fit = len(peaks) >= 20 and np.ptp(peaks) > 0
        for name, th in thresholds.items():
            if th['status'] != 'ok':
                assert m['operating_points'][name] is None
                if can_fit:
                    # Failed fit can also arise from support or finite-value checks;
                    # retain the recorded concrete reason instead of fabricating F1.
                    assert th.get('reason')
                continue
            if name == 'quantile':
                value = float(np.quantile(cal, .99))
            else:
                assert can_fit
                shape, _, scale = genpareto.fit(peaks, floc=0)
                equal(th['shape'], shape); equal(th['scale'], scale)
                value = u+genpareto.isf(.01*len(cal)/len(peaks), shape, loc=0, scale=scale)
            equal(th['threshold'], value)
            alarm = s >= value
            tp = int(((y == 1) & alarm).sum()); fp = int(((y == 0) & alarm).sum())
            fn = int(((y == 1) & ~alarm).sum()); tn = int(((y == 0) & ~alarm).sum())
            op = m['operating_points'][name]
            for k, v in [('tp', tp), ('fp', fp), ('fn', fn), ('tn', tn),
                         ('precision', tp/(tp+fp) if tp+fp else 0.), ('recall', tp/(tp+fn) if tp+fn else 0.),
                         ('f1', 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.), ('fpr', fp/(fp+tn)), ('alarm_fraction', np.mean(alarm))]:
                equal(op[k], v)
        verified.append(dict(track=track, dataset=task, method=method, seed=seed, test_rows=len(score),
                             test_target_sha256=sha(label_file), paired_baseline='ae',
                             reload_max_abs=r['checks']['checkpoint_max_abs']))
    expected = {(track, task, method, seed)
                for track, tasks, method in [('graph', ['banksim', 's_ffsd'], 'huge_financial_inductive'),
                   ('tabular', ['banksim', 's_ffsd'], 'himvh_zero_reconstruction'),
                   ('tabular', ['sapienza_5S', 'sapienza_15S', 'sapienza_25S', 'sd_stock', 'sd_crypto', 'sd_synth'], 'sdfmm_zero_synthetic')]
                for task in tasks for seed in [42, 43, 44]}
    assert seen == expected
    report = dict(status='passed', runs=len(records), operating_records=2*len(records), input_files=len(inputs),
                  checks=['pinned_scientific_source', 'input_bytes_SHA256', 'checkpoint_and_score_SHA256', 'prelabel_save_gate',
                          'same_baseline_training_calibration_test_indices', 'same_baseline_test_targets', 'train_only_common_scaler',
                          'unknown_mask', 'AP', 'trapezoid_PR_AUC', 'ROC_AUC', 'refitted_POT', 'q99', 'confusion_counts_and_rates'],
                  verified=verified)
    (args.runs/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('VERIFIED 30/30 runs, 60 operating records; paired cohorts and metrics passed')


if __name__ == '__main__':
    main()
