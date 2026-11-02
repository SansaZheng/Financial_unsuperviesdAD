"""Bounded-memory numeric cache; fitting can run without any label files."""
import hashlib
import json
from pathlib import Path

import numpy as np

DATASETS = ('sapienza_15S', 'tstr', 'sd_stock', 'sd_crypto', 'ulb', 'elliptic',
            'paysim', 'tabformer', 'ibm_aml', 'ieee_cis')
SPLITS = ('train', 'calibration', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


class Cache:
    def __init__(self, root, dataset):
        self.path = Path(root) / dataset
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        self.features = self.manifest['features']
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError('Need unique, nonempty numeric feature names')

    def x(self, split):
        n, p = self.manifest['splits'][split]['rows'], len(self.features)
        path = self.path / f'{split}.x.f32'
        if path.stat().st_size != n*p*4 or n < 1:
            raise ValueError('Feature cache length differs from manifest')
        return np.memmap(path, mode='r', dtype=np.float32, shape=(n, p))

    def y(self, split='test'):
        n = self.manifest['splits'][split]['rows']
        path = self.path / f'{split}.y.i8'
        if path.stat().st_size != n:
            raise ValueError('Target cache length differs from manifest')
        y = np.memmap(path, mode='r', dtype=np.int8, shape=(n,))
        if not np.isin(y, [-1, 0, 1]).all():
            raise ValueError('Targets must be 0=normal,1=anomaly,-1=unknown')
        return y


def sample(x, cap, seed):
    if cap < 1:
        raise ValueError('Sample cap must be positive')
    indices = np.sort(np.random.default_rng(seed).choice(len(x), min(cap, len(x)), replace=False))
    return np.asarray(x[indices]), indices
