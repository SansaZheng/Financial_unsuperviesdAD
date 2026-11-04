"""Check private sampling arrays against v1; publish only aggregate evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from finad.data import sha256
from finad.report import collect


def audit(reference, runs):
    _, records = collect(runs)
    checked = []
    for r in records:
        name = f'{r["method"]}_seed{r["seed"]}'
        actual = Path(runs) / r['dataset'] / name
        old = Path(reference) / r['dataset'] / f'iforest_seed{r["seed"]}'
        for split in ('train', 'calibration'):
            filename = f'{split}_indices.npy'
            np.testing.assert_array_equal(np.load(actual / filename), np.load(old / filename))
            assert sha256(actual / filename) == sha256(old / filename)
            checked.append(dict(dataset=r['dataset'], method=r['method'], seed=r['seed'],
                                split=split, sha256=sha256(actual / filename)))
    return dict(verified_runs=len(records), sampling_files_checked=len(checked),
                exact_indices_and_file_hash_match_v1=True, indices_published=False, checks=checked)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-runs', required=True, type=Path)
    p.add_argument('--runs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = audit(args.reference_runs, args.runs)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Exact v1 sampling comparison:', result['verified_runs'], 'runs,', result['sampling_files_checked'], 'files')
