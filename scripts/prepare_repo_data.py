"""Build label-blind feature caches directly from this repository's data/raw.

Ten tasks come from the eight downloaded source packages:

  ulb            ULB/Worldline credit card (transaction level)
  banksim        BankSim payment simulator (transaction level)
  s_ffsd         S-FFSD (transaction level; label 2 kept as unknown -1)
  sapienza_5S/15S/25S   Sapienza pump-and-dump chunk features (per-event time series)
  sd_stock / sd_crypto  SD-FMM publisher arrays (160-step point-level sequences)
  sd_synth       SD-FMM publisher synthetic sequences (all_DTWsim; synthetic anomalies)
  cardsim        one fixed-seed run of the shipped Federal Reserve CardSim simulator (synthetic)

Tasks that already existed in results/v1 (ulb, sapienza_15S, sd_stock, sd_crypto)
reuse exactly the v1 split rules and feature definitions, so the deterministic
group split (seed string ``kronosad-supervised-v1``) is reproduced here without
the external KronosAD loader. Targets are written to separate files and never
read while building features. Time-series tasks additionally store the test
sequence id of each row (``test.seq.i32``) so that sequence-aware metrics can be
computed later; rows of one sequence are contiguous and time ordered.

Not prepared: SynthAML (alert table only, no features) and ADMA21 (anomaly
intervals only, no market data). The CardSim transactions are generated here from
the shipped simulator with fixed parameters; the generated CSV stays local.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
SPLITS = ('train', 'calibration', 'test')
SPLIT_SEED = 'kronosad-supervised-v1'
SAPIENZA_FEATURES = ['std_rush_order', 'avg_rush_order', 'std_trades', 'std_volume', 'avg_volume',
                     'std_price', 'avg_price', 'avg_price_max', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
DATASETS = ('ulb', 'banksim', 's_ffsd', 'sapienza_5S', 'sapienza_15S', 'sapienza_25S',
            'sd_stock', 'sd_crypto', 'sd_synth', 'cardsim')
# Prespecified, label-free temporal tolerances (in rows) for sequence-aware metrics.
# Sapienza: the publisher states the labelled chunk may precede the announced time by up to
# 120 s, so 120 s / chunk length. SD-FMM: 5% of the fixed 160-step window.
TOLERANCE = {'sapienza_5S': 24, 'sapienza_15S': 8, 'sapienza_25S': 5, 'sd_stock': 8, 'sd_crypto': 8, 'sd_synth': 8}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def signed_log(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x)*np.log1p(np.abs(x))


def hash_onehot(values, bins=16):
    """Stable nominal hashing, independent of labels and partitions (same rule as v1)."""
    strings = pd.Series(values).fillna('<missing>').astype(str)
    mapping = {v: int.from_bytes(hashlib.sha256(v.encode()).digest()[:4], 'little') % bins for v in strings.unique()}
    return np.eye(bins, dtype=np.float32)[strings.map(mapping).to_numpy()]


def fixed_onehot(values, vocabulary):
    strings = pd.Series(values).astype(str)
    index = {v: i for i, v in enumerate(vocabulary)}
    out = np.zeros((len(strings), len(vocabulary)), dtype=np.float32)
    known = strings.map(index)
    out[np.arange(len(strings))[known.notna()], known.dropna().astype(int)] = 1.
    return out


def temporal_split(time, first, second):
    time = np.asarray(time)
    return np.where(time <= first, 0, np.where(time <= second, 1, 2)).astype(np.int8)


def group_splits(keys):
    """Deterministic 60/20/20 split of independent group IDs (identical rule to v1)."""
    keys = sorted(set(map(str, keys)), key=lambda k: hashlib.sha256(f'{SPLIT_SEED}:{k}'.encode()).hexdigest())
    train_end, validation_end = int(len(keys)*.6), int(len(keys)*.8)
    return {key: (0 if i < train_end else 1 if i < validation_end else 2) for i, key in enumerate(keys)}


class Writer:
    def __init__(self, root, name, protocol, extra=None):
        self.path = Path(root)/name
        self.path.mkdir(parents=True, exist_ok=False)
        self.meta = {'dataset': name, 'protocol': protocol, 'splits': {}}
        self.meta.update(extra or {})
        self.handles = {s: [(self.path/f'{s}.{ext}').open('wb') for ext in ('x.f32', 'y.i8', 'seq.i32')] for s in SPLITS}
        for s in SPLITS:
            self.meta['splits'][s] = {'rows': 0, 'positive': 0, 'normal': 0, 'unknown': 0, 'sequences': 0}
        self.last_sequence = {s: None for s in SPLITS}

    def append(self, x, y, split, names, sequence=None):
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int8)
        if x.ndim != 2 or len(x) != len(y) or len(names) != x.shape[1]:
            raise ValueError('Invalid feature/target shape')
        if not np.isin(y, [-1, 0, 1]).all() or not np.isfinite(x).all():
            raise ValueError('Invalid targets or non-finite features')
        if 'features' in self.meta and self.meta['features'] != list(names):
            raise ValueError('Feature schema changed between chunks')
        self.meta['features'] = list(names)
        split = np.broadcast_to(np.asarray(split), (len(x),))
        sequence = np.full(len(x), -1, dtype=np.int32) if sequence is None else np.asarray(sequence, dtype=np.int32)
        for i, name in enumerate(SPLITS):
            mask = split == i
            if not mask.any():
                continue
            x[mask].tofile(self.handles[name][0]); y[mask].tofile(self.handles[name][1])
            sequence[mask].tofile(self.handles[name][2])
            counts = self.meta['splits'][name]
            counts['rows'] += int(mask.sum())
            for label, key in ((1, 'positive'), (0, 'normal'), (-1, 'unknown')):
                counts[key] += int((y[mask] == label).sum())
            seq = sequence[mask]
            if (seq >= 0).any():
                changes = int(np.count_nonzero(np.diff(seq)))+1
                if self.last_sequence[name] is not None and self.last_sequence[name] == seq[0]:
                    changes -= 1
                counts['sequences'] += changes
                self.last_sequence[name] = int(seq[-1])

    def close(self):
        for handles in self.handles.values():
            for h in handles:
                h.close()
        if any(not v['rows'] for v in self.meta['splits'].values()):
            raise ValueError(f'Empty partition: {self.meta}')
        if not self.meta.get('temporal'):
            for s in SPLITS:
                (self.path/f'{s}.seq.i32').unlink()
                del self.meta['splits'][s]['sequences']
        (self.path/'manifest.json').write_text(json.dumps(self.meta, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({'prepared': self.meta['dataset'], 'features': len(self.meta['features']),
                          'splits': self.meta['splits']}), flush=True)


def read_zip_csv(zip_path, member, **kwargs):
    with zipfile.ZipFile(zip_path) as archive:
        data = archive.read(member)  # whole member in memory; chunked parsing stays valid after the archive closes
    return pd.read_csv(io.BytesIO(data), **kwargs)


def prepare_ulb(root):
    w = Writer(root, 'ulb', 'Time<=86400 train; <=129600 calibration; later test. Time/label excluded from features.',
               {'source': 'data/raw/ulb_creditcard/ulb_creditcard.zip'})
    names = [f'V{i}' for i in range(1, 29)]+['log_Amount']
    for d in read_zip_csv(RAW/'ulb_creditcard/ulb_creditcard.zip', 'creditcard.csv', chunksize=100000):
        x = np.column_stack([d[[f'V{i}' for i in range(1, 29)]].to_numpy(), signed_log(d.Amount)])
        w.append(x, d.Class.to_numpy(), temporal_split(d.Time, 86400, 129600), names)
    w.close()


def cumulative_time_boundaries(times):
    counts = pd.Series(times).value_counts().sort_index()
    cumulative = counts.cumsum().to_numpy()
    return [counts.index[np.searchsorted(cumulative, cumulative[-1]*fraction)] for fraction in (.6, .8)]


def prepare_banksim(root):
    d = read_zip_csv(RAW/'banksim/banksim.zip', 'bs140513_032310.csv')
    for c in ('customer', 'age', 'gender', 'zipcodeOri', 'merchant', 'zipMerchant', 'category'):
        d[c] = d[c].astype(str).str.strip("'")
    bounds = cumulative_time_boundaries(d.step)
    w = Writer(root, 'banksim', 'Chronological 60/20/20 by cumulative time-only row counts over whole simulation '
               'steps. No customer ID; merchant hashed; constant zip codes dropped; label never read for features.',
               {'source': 'data/raw/banksim/banksim.zip:bs140513_032310.csv', 'time_boundaries': [int(b) for b in bounds]})
    ages, genders = ['0', '1', '2', '3', '4', '5', '6', 'U'], ['F', 'M', 'E', 'U']
    x = np.column_stack([signed_log(d.amount), fixed_onehot(d.age, ages), fixed_onehot(d.gender, genders),
                         hash_onehot(d.category, 32), hash_onehot(d.merchant, 16)])
    names = ['log_amount']+[f'age_{a}' for a in ages]+[f'gender_{g}' for g in genders]+\
            [f'category_hash_{i}' for i in range(32)]+[f'merchant_hash_{i}' for i in range(16)]
    w.append(x, d.fraud.to_numpy(), temporal_split(d.step, *bounds), names)
    w.close()


def prepare_s_ffsd(root):
    d = read_zip_csv(RAW/'s_ffsd/data/S-FFSD.zip', 'S-FFSD.csv')
    if not d.Time.is_monotonic_increasing:
        raise ValueError('S-FFSD rows are expected in time order')
    bounds = cumulative_time_boundaries(d.Time)
    w = Writer(root, 's_ffsd', 'Chronological 60/20/20 by cumulative Time counts. Label 2 (unlabelled) kept as '
               'unknown (-1) during fitting/calibration and masked only for evaluation. No Source account IDs.',
               {'source': 'data/raw/s_ffsd/data/S-FFSD.zip:S-FFSD.csv', 'time_boundaries': [int(b) for b in bounds]})
    x = np.column_stack([signed_log(d.Amount), hash_onehot(d.Location, 16), hash_onehot(d.Type, 16), hash_onehot(d.Target, 16)])
    names = ['log_amount']+[f'location_hash_{i}' for i in range(16)]+[f'type_hash_{i}' for i in range(16)]+[f'target_hash_{i}' for i in range(16)]
    y = np.where(d.Labels.to_numpy() == 2, -1, d.Labels.to_numpy()).astype(np.int8)
    w.append(x, y, temporal_split(d.Time, *bounds), names)
    w.close()


def prepare_sapienza(root, resolution):
    tables = {r: pd.read_csv(RAW/'sapienza_pump_dump/labeled_features'/f'features_{r}.csv.gz') for r in ('5S', '15S', '25S')}
    # A coin cannot occur in multiple partitions, including across resolutions (same rule as v1).
    mapping = group_splits({symbol for t in tables.values() for symbol in t.symbol})
    d = tables[resolution]
    if d.groupby('pump_index').symbol.nunique().max() != 1 or not d.pump_index.is_monotonic_increasing:
        raise ValueError('A pump_index must identify one coin and rows must be grouped by event')
    dates = pd.to_datetime(d.date, utc=True)
    if (dates.groupby(d.pump_index).diff().dt.total_seconds().dropna() <= 0).any():
        raise ValueError('Rows must be time ordered within each event')
    w = Writer(root, f'sapienza_{resolution}', 'Existing coin-group split; same event/coin never split by row.',
               {'source': f'data/raw/sapienza_pump_dump/labeled_features/features_{resolution}.csv.gz',
                'temporal': True, 'sequence_unit': 'pump_index', 'sequence_id_file': 'test.seq.i32',
                'tolerance_rows': TOLERANCE[f'sapienza_{resolution}'],
                'tolerance_note': 'Publisher states the labelled chunk may precede the announced pump by up to 120 s; '
                                  'rows are not uniformly spaced in time (empty chunks are absent).'})
    split = d.symbol.map(mapping).to_numpy()
    w.append(d[SAPIENZA_FEATURES].to_numpy(dtype=np.float64), d['gt'].to_numpy(), split, SAPIENZA_FEATURES, d.pump_index.to_numpy())
    w.close()


def point_features(x):
    """Past/current values only within each sequence; identical to the v1 definition."""
    x = np.asarray(x, dtype=np.float64)[..., 0]
    parts = [x, np.diff(x, prepend=x[:, :1], axis=1)]
    names = ['publisher_value', 'change']
    frame = pd.DataFrame(x.T)
    for width in (4, 16, 32):
        roll = frame.rolling(width, min_periods=1)
        parts.extend([roll.mean().to_numpy().T, roll.std(ddof=0).to_numpy().T, roll.max().to_numpy().T-roll.min().to_numpy().T])
        names.extend([f'mean_{width}', f'std_{width}', f'range_{width}'])
    return np.stack(parts, -1).reshape(-1, len(parts)).astype(np.float32), names


def prepare_sd(root, name):
    if name == 'sd_stock':
        base, suffix, protocol = RAW/'sd_fmm/stock cases', '_bs', 'Existing deduplicated sequence-group split; offline retrospective task.'
    elif name == 'sd_crypto':
        base, suffix, protocol = RAW/'sd_fmm/cryptocurrency cases', '', 'Existing deduplicated sequence-group split; offline retrospective task.'
    else:
        base, suffix, protocol = RAW/'sd_fmm/systhesis', None, ('Deterministic hash-group 60/20/20 split of the publisher '
                                                               'synthetic sequences (all_DTWsim); synthetic anomalies, not real manipulation.')
    if suffix is None:
        x = np.load(base/'all_DTWsim_data.npy', allow_pickle=False); y = np.load(base/'all_DTWsim_label.npy', allow_pickle=False)
    else:
        x = np.load(base/f'test_data{suffix}.npy', allow_pickle=False); y = np.load(base/f'test_label{suffix}.npy', allow_pickle=False)
    if x.shape != (*y.shape, 1) or y.shape[1] != 160 or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f'Invalid SD-FMM arrays for {name}')
    hashes = [hashlib.sha256(a.astype('<f8').tobytes()).hexdigest() for a in x]
    mapping = group_splits(hashes)
    split = np.repeat([mapping[h] for h in hashes], 160)
    features, names = point_features(x)
    w = Writer(root, name, protocol, {'source': str(base.relative_to(ROOT)), 'temporal': True, 'sequence_unit': 'sequence_index',
                                      'sequence_id_file': 'test.seq.i32', 'sequence_length': 160, 'tolerance_rows': TOLERANCE[name]})
    w.append(features, y.reshape(-1).astype(np.int8), split, names, np.repeat(np.arange(len(x)), 160))
    w.close()


CARDSIM = {'seed': 42, 'n_payers': 4000, 'n_days': 180}


def prepare_cardsim(root):
    """Generate one fixed CardSim run from the shipped official simulator, then featurize it label-blind."""
    import importlib
    import sys
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(RAW/'cardsim/feds-2025017-python.zip') as archive:
            archive.extractall(tmp)
        package_root = Path(tmp)/'cardsim'
        sys.path.insert(0, str(package_root))
        try:
            cardsim = importlib.import_module('cardsim')
            simulator = cardsim.Cardsim(seed=CARDSIM['seed'], dcpc_folder=package_root/'cardsim/dcpc')
            d = simulator.simulate(n_payers=CARDSIM['n_payers'], n_days=CARDSIM['n_days'])
        finally:
            sys.path.remove(str(package_root))
            for module in [m for m in sys.modules if m == 'cardsim' or m.startswith('cardsim.')]:
                del sys.modules[module]
    d = d.sort_values(['date_time', 'payer_id', 'payee_id'], kind='stable').reset_index(drop=True)
    generated = ROOT/'.local_data'/'cardsim_generated.csv'
    generated.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(generated, index=False)
    bounds = cumulative_time_boundaries(d.day_index)
    w = Writer(root, 'cardsim', 'Fixed-parameter run of the shipped Federal Reserve CardSim simulator (synthetic). '
               'Chronological 60/20/20 by cumulative counts over whole simulated days. No payer/payee IDs.',
               {'source': 'data/raw/cardsim/feds-2025017-python.zip', 'simulator_parameters': CARDSIM,
                'generated_rows': int(len(d)), 'generated_sha256': sha256_file(generated),
                'time_boundaries': [int(b) for b in bounds], 'synthetic': True})
    seconds = d.time_seconds.to_numpy(dtype=np.float64)
    weekday = pd.to_datetime(d.date).dt.weekday.to_numpy()
    x = np.column_stack([signed_log(d.amount), d.credit_card, d.remote, signed_log(d.distance),
                         np.sin(seconds*2*np.pi/86400), np.cos(seconds*2*np.pi/86400), np.eye(7, dtype=np.float32)[weekday]])
    names = ['log_amount', 'credit_card', 'remote', 'log_distance', 'tod_sin', 'tod_cos']+[f'weekday_{i}' for i in range(7)]
    w.append(x, d.fraud.to_numpy(), temporal_split(d.day_index, *bounds), names)
    w.close()


def prepare(name, root):
    if name == 'cardsim':
        prepare_cardsim(root)
    elif name == 'ulb':
        prepare_ulb(root)
    elif name == 'banksim':
        prepare_banksim(root)
    elif name == 's_ffsd':
        prepare_s_ffsd(root)
    elif name.startswith('sapienza_'):
        prepare_sapienza(root, name.split('_')[1])
    else:
        prepare_sd(root, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if (args.output/name/'manifest.json').exists():
            print(f'Already prepared: {name}', flush=True)
            continue
        try:
            prepare(name, args.output)
        except BaseException:
            # Never leave a half-written cache behind: a later run would treat it as prepared.
            import shutil
            shutil.rmtree(args.output/name, ignore_errors=True)
            raise
    hashes = {str(p.relative_to(args.output)): {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
              for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'cache_files_sha256.json'}
    (args.output/'cache_files_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
