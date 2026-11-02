"""Seven classical inductive detectors. No fit method accepts target labels."""
import warnings
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

METHODS = ('iforest', 'lof', 'knn', 'ocsvm', 'pca', 'hbos', 'gmm')


class ClassicalDetector:
    def __init__(self, method, seed=42):
        if method not in METHODS:
            raise ValueError(f'Unknown classical method {method}')
        self.method, self.seed = method, seed

    def fit(self, z):
        z = np.asarray(z, dtype=np.float32)
        self.info = {'method': self.method}
        k = min(20, len(z)-1)
        if len(z) < 2:
            raise ValueError('Need >=2 training points')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            if self.method == 'iforest':
                self.model = IsolationForest(n_estimators=100, max_samples=min(256, len(z)),
                    contamination='auto', random_state=self.seed, n_jobs=2).fit(z)
                self.info.update(n_estimators=100, max_samples=min(256, len(z)))
            elif self.method == 'lof':
                self.model = LocalOutlierFactor(n_neighbors=k, novelty=True, algorithm='brute',
                    n_jobs=2, contamination='auto').fit(z)
                self.info.update(n_neighbors=k, mode='inductive_novelty')
            elif self.method == 'knn':
                self.model = NearestNeighbors(n_neighbors=k, algorithm='brute', n_jobs=2).fit(z)
                self.info.update(n_neighbors=k, distance='euclidean', aggregation='kth')
            elif self.method == 'ocsvm':
                self.model = OneClassSVM(kernel='rbf', nu=.05, gamma='scale', cache_size=256).fit(z)
                self.info.update(kernel='rbf', nu=.05, gamma=float(self.model._gamma),
                                 fit_status=int(self.model.fit_status_), support_vectors=len(self.model.support_))
            elif self.method == 'pca':
                self.center = z.mean(0)
                if z.shape[1] == 1 or np.var(z, axis=0).sum() == 0:
                    self.components = np.empty((0, z.shape[1]), dtype=np.float32)
                else:
                    self.model = PCA(n_components=.95, svd_solver='full').fit(z)
                    keep = min(self.model.n_components_, z.shape[1]-1)
                    self.center, self.components = self.model.mean_, self.model.components_[:keep]
                self.info.update(retained_variance_target=.95, retained_components=len(self.components),
                                 score='squared_prediction_error_sum')
            elif self.method == 'hbos':
                self.histograms = []
                for j in range(z.shape[1]):
                    counts, edges = np.histogram(z[:, j], bins=10)
                    self.histograms.append((counts, edges, int(counts.max())))
                self.info.update(bins=10, binning='equal_width', pseudocount=1,
                                 density_normalization='maximum_per_feature')
            else:
                # Diagonal variance uses E[x^2]-E[x]^2: float32 cancellation
                # on sparse/near-constant financial features can make it negative.
                # Keep fixed components/reg_covar, use float64 EM arithmetic.
                self.model = GaussianMixture(n_components=min(4, len(z)), covariance_type='diag',
                    random_state=self.seed, reg_covar=1e-4, max_iter=200, n_init=1).fit(z.astype(np.float64))
                self.info.update(n_components=self.model.n_components, covariance_type='diag',
                                 reg_covar=1e-4, converged=bool(self.model.converged_),
                                 iterations=int(self.model.n_iter_), fit_dtype='float64')
            self.info['warnings'] = sorted(set(str(w.message) for w in captured))
        return self

    def score(self, z):
        if self.method in ('iforest', 'lof'):
            return -self.model.score_samples(z)
        if self.method == 'knn':
            return self.model.kneighbors(z, return_distance=True)[0][:, -1]
        if self.method == 'ocsvm':
            return -self.model.decision_function(z)
        if self.method == 'pca':
            centered = z-self.center
            reconstructed = (centered @ self.components.T) @ self.components
            return np.sum((centered.astype(np.float64)-reconstructed)**2, axis=1)
        if self.method == 'hbos':
            scores = np.zeros(len(z), dtype=np.float64)
            for j, (counts, edges, maximum) in enumerate(self.histograms):
                bins = np.searchsorted(edges, z[:, j], side='right')-1
                bins[z[:, j] == edges[-1]] = len(counts)-1
                inside = (z[:, j] >= edges[0]) & (z[:, j] <= edges[-1])
                observed = np.zeros(len(z), dtype=np.float64)
                observed[inside] = counts[bins[inside]]
                scores += np.log(maximum+1)-np.log(observed+1)
            return scores
        return -self.model.score_samples(z)  # Gaussian negative log density can be signed.

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        # Load only checkpoints you created/trust: joblib uses pickle.
        return joblib.load(path)
