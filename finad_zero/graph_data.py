"""Actual observed payment graphs, no label-dependent edges or invented device links.

Each phase is an offline four-snapshot window. Role-typed source and target IDs
remain distinct, including S-FFSD whose two identifier sets do not overlap.
Unknown incident transactions prevent a negative account label (positive wins).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_repo_data import RAW, read_zip_csv, cumulative_time_boundaries, temporal_split, sha256_file

SPLITS = ('train', 'calibration', 'test')
FEATURES = ['source_role', 'target_role', 'log_out_count', 'log_in_count',
            'out_mean_log_amount', 'in_mean_log_amount', 'out_std_log_amount', 'in_std_log_amount',
            'out_max_log_amount', 'in_max_log_amount']


def aggregate_targets(src, dst, labels, n):
    pos = np.zeros(n, bool); unknown = np.zeros(n, bool); seen = np.zeros(n, bool)
    for endpoint in (src, dst):
        seen[endpoint] = True; pos[endpoint[labels == 1]] = True
        unknown[endpoint[labels == -1]] = True
    return np.where(pos, 1, np.where(unknown | ~seen, -1, 0)).astype(np.int8)


def make_phase(src_ids, dst_ids, time, amount):
    """Features/topology only; this function cannot receive a target array."""
    ids = sorted(set('s:'+str(v) for v in src_ids) | set('t:'+str(v) for v in dst_ids))
    index = {key: i for i, key in enumerate(ids)}
    src = np.array([index['s:'+str(v)] for v in src_ids]); dst = np.array([index['t:'+str(v)] for v in dst_ids])
    # Whole timestamps never straddle snapshot boundaries. Quantiles use time only.
    counts = pd.Series(time).value_counts().sort_index()
    boundaries = [counts.index[min(np.searchsorted(counts.cumsum(), counts.sum()*q), len(counts)-1)]
                  for q in (.25, .5, .75)]
    snap = np.searchsorted(boundaries, np.asarray(time), side='left')
    x = np.zeros((4, len(ids), len(FEATURES)), dtype=np.float32)
    active = np.zeros((4, len(ids)), bool); edges = []
    value = np.sign(amount)*np.log1p(np.abs(amount))
    for t in range(4):
        mask = snap == t; a, b, v = src[mask], dst[mask], value[mask]
        pairs = np.unique(np.column_stack([a, b]), axis=0)
        edges.append(pairs.T.astype(np.int64))
        active[t, a] = True; active[t, b] = True
        x[t, :, 0] = [k.startswith('s:') for k in ids]
        x[t, :, 1] = 1-x[t, :, 0]
        for direction, endpoint in enumerate((a, b)):
            count = np.bincount(endpoint, minlength=len(ids))
            total = np.bincount(endpoint, weights=v, minlength=len(ids))
            squares = np.bincount(endpoint, weights=v*v, minlength=len(ids))
            mean = total/np.maximum(count, 1)
            maximum = np.zeros(len(ids)); np.maximum.at(maximum, endpoint, v)
            x[t, :, 2+direction] = np.log1p(count)
            x[t, :, 4+direction] = mean
            x[t, :, 6+direction] = np.sqrt(np.maximum(squares/np.maximum(count, 1)-mean*mean, 0))
            x[t, :, 8+direction] = maximum
    return x, edges, active, src, dst, ids, boundaries


def prepare(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for name, path, member, columns in (
        ('banksim', RAW/'banksim/banksim.zip', 'bs140513_032310.csv', ['customer','merchant','step','amount','fraud']),
        ('s_ffsd', RAW/'s_ffsd/data/S-FFSD.zip', 'S-FFSD.csv', ['Source','Target','Time','Amount','Labels'])):
        d = read_zip_csv(path, member); a,b,t,v,y = columns
        for c in (a,b): d[c] = d[c].astype(str).str.strip("'")
        bounds = cumulative_time_boundaries(d[t]); split = temporal_split(d[t], *bounds)
        out = root/name; out.mkdir(exist_ok=False)
        meta = {'dataset': name, 'unit': 'role-typed account active during phase', 'features': FEATURES,
                'time_boundaries': list(map(float,bounds)), 'raw_sha256': sha256_file(path), 'splits': {}}
        for k,s in enumerate(SPLITS):
            frame = d.loc[split == k]
            # Fixed label-blind graph budget: up to 4096 paying accounts, all
            # their counterparties and transactions. Do not fabricate kNN edges.
            sources = sorted(frame[a].unique(), key=lambda key: hashlib.sha256(
                ('financial-zero-20261114:'+str(key)).encode()).hexdigest())[:4096]
            frame = frame.loc[frame[a].isin(sources)]
            x, edges, active, src, dst, ids, sb = make_phase(frame[a], frame[b], frame[t], frame[v].to_numpy())
            np.savez_compressed(out/f'{s}.graph.npz', x=x, active=active, **{f'e{i}':e for i,e in enumerate(edges)})
            (out/f'{s}.ids.json').write_text(json.dumps(ids))
            # Target aggregation occurs only AFTER all features/edges are fixed.
            labels = frame[y].to_numpy(); labels = np.where(labels == 2, -1, labels)
            target = aggregate_targets(src,dst,labels,len(ids)); np.save(out/f'{s}.y.npy', target, allow_pickle=False)
            meta['splits'][s] = {'nodes':len(ids), 'edges':sum(e.shape[1] for e in edges),
                                'evaluated_source_accounts':len(sources),
                                'positive':int((target==1).sum()), 'normal':int((target==0).sum()),
                                'unknown':int((target==-1).sum()), 'snapshot_boundaries':list(map(float,sb))}
        meta['graph_budget'] = '4096 SHA256-selected paying accounts per phase; retain all counterparties; evaluate payers only'
        (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
        print(json.dumps(meta['splits']), flush=True)


def load_phase(path, split):
    with np.load(Path(path)/f'{split}.graph.npz', allow_pickle=False) as z:
        return z['x'], [z[f'e{i}'] for i in range(4)], z['active']


def induce(x, edges, active, indices):
    inverse = np.full(x.shape[1], -1, dtype=np.int64); inverse[indices] = np.arange(len(indices))
    new_edges = []
    for e in edges:
        e = inverse[e]; new_edges.append(e[:, (e >= 0).all(0)])
    return x[:, indices], new_edges, active[:, indices]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
