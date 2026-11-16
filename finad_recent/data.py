"""Label-blind native transaction keys and checks of the frozen input files."""
import hashlib
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_repo_data import RAW, read_zip_csv, temporal_split, cumulative_time_boundaries
from .himvh import hyperedges

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('train', 'calibration', 'test')
TEMPORAL = ('sapienza_5S', 'sapienza_15S', 'sapienza_25S', 'sd_stock', 'sd_crypto', 'sd_synth')
GRAPH = ('banksim', 's_ffsd')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_inputs(tabular, graphs):
    checked = {}
    golden = json.loads((ROOT/'results/zero_label_20261114/input_cache_files_sha256.json').read_text())
    for name, entry in golden.items():
        p = tabular/name
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('Paired tabular input mismatch: '+name)
        checked['tabular/'+name] = entry
    golden = json.loads((ROOT/'docs/recent_zero_graph_inputs.json').read_text())
    for name, entry in golden.items():
        p = graphs/name
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('Paired graph input mismatch: '+name)
        checked['graphs/'+name] = entry
    return checked


def transaction_keys(dataset, phase, indices=None):
    """Read only identity/time/attribute columns, never open a target column."""
    if dataset == 'banksim':
        columns = ['customer', 'merchant', 'category', 'gender', 'step']
        d = read_zip_csv(RAW/'banksim/banksim.zip', 'bs140513_032310.csv', usecols=columns)
        time = 'step'
    elif dataset == 's_ffsd':
        columns = ['Source', 'Target', 'Location', 'Type', 'Time']
        d = read_zip_csv(RAW/'s_ffsd/data/S-FFSD.zip', 'S-FFSD.csv', usecols=columns)
        time = 'Time'
    else:
        raise ValueError('No native transaction views for '+dataset)
    bounds = cumulative_time_boundaries(d[time]); split = temporal_split(d[time], *bounds)
    d = d.loc[split == PHASES.index(phase)]
    if indices is not None:
        d = d.iloc[indices]
    # Sort view members by time, breaking ties by original phase row order.
    order = np.argsort(d[time].to_numpy(), kind='stable')
    views = []
    for c in columns[:-1]:
        key = d[c].astype(str).str.strip("'").to_numpy()
        e = hyperedges(key[order], 4)
        views.append(order[e])
    return views, len(d)
