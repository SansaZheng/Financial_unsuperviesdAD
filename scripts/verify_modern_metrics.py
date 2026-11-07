"""Independently recompute a sample of the modern metrics from the saved scores and labels.

    python scripts/verify_modern_metrics.py --runs .local_data/runs_A .local_data/runs_B \
        --cache .local_data/feature_cache --output results/data_2026/verification_modern.json

For every sampled run the pooled AUC-PR/AUC-ROC are recomputed with scikit-learn straight from
the score file, and for temporal tasks the per-sequence VUS-PR, VUS-ROC and Affiliation-F are
recomputed with TSB-AD's own ``get_metrics`` entry point (a different code path from
``finad.modern_metrics``) and macro-averaged; all values must agree with the cached
``modern_metrics.json`` to 1e-9.
"""
import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finad.data import Cache, sha256  # noqa: E402
from finad.experiment import write_json  # noqa: E402


def recompute(run, cache_root):
    from TSB_AD.evaluation.metrics import get_metrics
    result = json.loads((run/'result.json').read_text())
    modern = json.loads((run/'modern_metrics.json').read_text())
    path = run/result['test_scores_file']
    assert sha256(path) == result['test_scores_sha256'] == modern['test_scores_sha256']
    if path.suffix == '.npz':
        with np.load(path) as z:
            scores = z['scores']
    else:
        scores = np.load(path)
    cache = Cache(cache_root, result['dataset'])
    y = np.asarray(cache.y('test'))
    known = y != -1
    checks = {'pooled_auc_pr': (float(average_precision_score(y[known], scores[known])), modern['point']['average_precision']),
              'pooled_auc_roc': (float(roc_auc_score(y[known], scores[known])), modern['point']['roc_auc'])}
    if cache.manifest.get('temporal'):
        seq = np.fromfile(cache.path/cache.manifest['sequence_id_file'], dtype=np.int32)
        tolerance = cache.manifest['tolerance_rows']
        pot = result['thresholds']['pot']
        values = {'vus_pr': [], 'vus_roc': [], 'affiliation_f1': [], 'pa_f1': []}
        cuts = np.concatenate([[0], np.flatnonzero(np.diff(seq))+1, [len(seq)]])
        for a, b in zip(cuts[:-1], cuts[1:]):
            label = y[a:b].astype(int)
            if label.sum() == 0 or label.all():
                continue
            pred = (scores[a:b] >= pot['threshold']).astype(int) if pot['status'] == 'ok' else np.zeros(b-a, int)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                official = get_metrics(scores[a:b], label, slidingWindow=2*tolerance, pred=pred)
            values['vus_pr'].append(official['VUS-PR']); values['vus_roc'].append(official['VUS-ROC'])
            if pot['status'] == 'ok':
                values['affiliation_f1'].append(0. if pred.sum() == 0 else official['Affiliation-F'])
                values['pa_f1'].append(official['PA-F1'])
        free = modern['sequence']['threshold_free']
        checks['seq_vus_pr'] = (float(np.mean(values['vus_pr'])), free['vus_pr'])
        checks['seq_vus_roc'] = (float(np.mean(values['vus_roc'])), free['vus_roc'])
        if pot['status'] == 'ok':
            op = modern['sequence']['operating_points']['pot']
            checks['seq_affiliation_f1_pot'] = (float(np.mean(values['affiliation_f1'])), op['affiliation_f1'])
            checks['seq_pa_f1_pot'] = (float(np.mean(values['pa_f1'])), op['pa_f1'])
    for name, (fresh, cached) in checks.items():
        if not np.isclose(fresh, cached, rtol=0, atol=1e-9):
            raise AssertionError(f'{run}: {name} recomputed {fresh} != cached {cached}')
    return {'dataset': result['dataset'], 'method': result['method'], 'seed': result['seed'], 'checks': sorted(checks)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-dataset', type=int, default=3, help='runs sampled per dataset (deterministic)')
    args = parser.parse_args()
    runs = sorted(p.parent for root in args.runs for p in root.glob('*/*/modern_metrics.json'))
    by_dataset = {}
    for run in runs:
        by_dataset.setdefault(run.parent.name, []).append(run)
    rng = np.random.default_rng(2026)
    sampled = [run for name, group in sorted(by_dataset.items())
               for run in sorted(rng.choice(group, min(args.per_dataset, len(group)), replace=False).tolist())]
    verified = [recompute(run, args.cache) for run in sampled]
    summary = {'runs_available': len(runs), 'runs_recomputed': len(verified), 'tolerance': 1e-9,
               'independent_code_paths': ['sklearn average_precision_score/roc_auc_score on the raw score file',
                                          'TSB_AD.evaluation.metrics.get_metrics per sequence, macro-averaged'],
               'runs': verified}
    write_json(args.output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}))
