"""Train-only robust preprocessing shared by all ten detectors."""
import warnings
import numpy as np


class Preprocessor:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or np.isinf(x).any():
            raise ValueError('Need at least two rows, a 2D matrix, no infinity')
        self.keep = np.isfinite(x).any(0)
        if not self.keep.any():
            raise ValueError('All training columns are missing')
        x = x[:, self.keep]
        self.missing = np.isnan(x).any(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.center = np.nanmedian(x, axis=0)
            q1, q3 = np.nanquantile(x, [.25, .75], axis=0)
            std = np.nanstd(x, axis=0)
        scale = (q3-q1)/1.349
        self.scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.))
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)[:, self.keep]
        if np.isinf(x).any():
            raise ValueError('Infinite input')
        missing = np.isnan(x)
        z = (np.where(missing, self.center, x)-self.center)/self.scale
        # Same bounded observations for training AND scoring of every detector.
        return np.column_stack([np.clip(z, -8, 8), missing[:, self.missing]]).astype(np.float32)

    def save(self, path):
        np.savez(path, keep=self.keep, missing=self.missing, center=self.center, scale=self.scale)

    @classmethod
    def load(cls, path):
        instance = cls()
        with np.load(path, allow_pickle=False) as data:
            for key in ('keep', 'missing', 'center', 'scale'):
                setattr(instance, key, data[key])
        return instance
