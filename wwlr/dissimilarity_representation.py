
import numpy as np
import ot
import warnings

from tqdm import tqdm
from typing import List, Tuple
from .utilities import sparse_histogram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_extra.cluster import KMedoids


class WassersteinDissimilarity(TransformerMixin, BaseEstimator):
    '''
    Class that computes the distribution representation as Wasserstein distances with a reference set
    '''

    def __init__(self, n_refs: int, categorical: bool = False, max_bins: int = 2000, approximate: bool = False,
                 regularization: float = 0.05, random_refs: bool = False, n_jobs: int = 1) -> None:
        self.n_refs = n_refs
        self.categorical = categorical
        self.max_bins = max_bins
        self.approximate = approximate
        self.regularization = regularization
        self.random_refs = random_refs
        self.n_jobs = n_jobs

    def wasserstein_distances(self, histograms: List[Tuple[np.ndarray, np.ndarray]],
                              ref_histograms: List[Tuple[np.ndarray, np.ndarray]], max_iter: int = 10000) -> np.ndarray:
        n = len(histograms)
        m = len(ref_histograms)
        M = np.zeros((n, m))

        hist_pairs = []
        for i in range(n):
            if histograms is ref_histograms:
                iterator = range(i+1, m)
            else:
                iterator = range(m)

            for j in iterator:
                hist_pairs.append((histograms[i], ref_histograms[j]))

        def calculate_dist(hists):
            # hist_i = tuple(count, middle point bin of h dimensional space)
            hist_i, hist_j = hists
            ground_distance = 'hamming' if self.categorical else 'euclidean'
            costs = ot.dist(hist_i[0], hist_j[0], metric=ground_distance)

            density_i = hist_i[1] / hist_i[1].sum()
            density_j = hist_j[1] / hist_j[1].sum()

            if self.approximate:
                return ot.sinkhorn2(density_i, density_j, costs, self.regularization, numItermax=max_iter)
            else:
                return ot.emd2(density_i, density_j, costs, numItermax=max_iter)

        if self.n_jobs > 1:
            try:
                # Uses pathos in the background
                from p_tqdm import p_map
                M = p_map(calculate_dist, hist_pairs, num_cpus=self.n_jobs)
            except Exception:
                from pathos.pools import ProcessPool
                p = ProcessPool(self.n_jobs)
                M = p.map(calculate_dist, hist_pairs)
        else:
            M = list(map(calculate_dist, tqdm(hist_pairs)))

        M = np.array(M)

        if histograms is ref_histograms:
            Mp = np.zeros((n, n))
            Mp[np.triu_indices(n, 1)] = M
            return (Mp + Mp.T)
        else:
            return M.reshape(n, m)

    def optimal_sparse_histogram(self, samples: np.ndarray, max_iter: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        bins, counts = np.unique(samples, return_counts=True, axis=0)
        if len(bins) < self.max_bins:
            return bins, counts

        n_bins = 1000000

        for i in range(max_iter):
            scaler = (0.75 * self.max_bins / len(bins) - 1) * (1 - i / max_iter) + 1
            n_bins = max(min(int(n_bins * scaler), 1000000), 2)
            bins, counts = sparse_histogram(samples, n_bins)

            if 0.5 * self.max_bins < len(bins) < self.max_bins:
                return bins, counts

        warnings.warn('optimal_sparse_histogram failed to converge. Consider increasing the number of iterations.')

        # failsafe
        if len(bins) > self.max_bins:

            n_bins = n_bins / 100
            bins, counts = sparse_histogram(samples, max(int(n_bins), 2))
            assert(len(bins) < self.max_bins)

        return bins, counts

    def create_histograms(self, X: List[np.ndarray], max_iter: int = 25) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.categorical:
            histograms = list(map(lambda x: np.unique(x, return_counts=True, axis=0), X))
        else:
            histograms = list(map(lambda x: self.optimal_sparse_histogram(x, max_iter=max_iter), X))
        return histograms

    def median_cluster_indices(self, distances: np.ndarray, max_iter: int = 300) -> np.ndarray:
        indices = np.empty((30, self.n_refs), dtype=int)
        for i in range(30):
            kmedoids = KMedoids(n_clusters=self.n_refs, metric='precomputed', max_iter=max_iter, init='k-medoids++')
            kmedoids.fit(distances)
            indices[i, :] = np.sort(kmedoids.medoid_indices_)

        _, counts = np.unique(indices, return_counts=True, axis=0)  # type:ignore
        return indices[np.argmax(counts)]

    def to_separate_hists(self, X: List[np.ndarray], bipartite_sets: List[np.ndarray] = [], hist_max_iter: int = 25
                          ) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
        if bipartite_sets:
            X_0 = [X[i][bs] for i, bs in enumerate(bipartite_sets)]
            X_1 = [np.delete(X[i], bs, axis=0) for i, bs in enumerate(bipartite_sets)]


            # X_1 = matrix row h=0 [10, 5, 12], row h=1 [9, 7, 10], row h....
            list_of_hists = [self.create_histograms(X_0, hist_max_iter), self.create_histograms(X_1, hist_max_iter)]
            # list_of_hist histogram per row (results in h histograms)

        else:
            list_of_hists = [self.create_histograms(X, hist_max_iter)]
        return list_of_hists

    def fit(self, X: List[np.ndarray], bipartite_sets: List[np.ndarray] = [],
            hist_max_iter: int = 25, emd_max_iter: int = 10000, cluster_max_iter: int = 300,
            reuse_distances: bool = False):
        # f: G -> R^... => X_G^h = [a^h(v_1), \ldots, a^h(v_{n_g})]^T => h=0 [10, 5, 12], h=1 [9, 7, 10], 
        list_of_hists = self.to_separate_hists(X, bipartite_sets, hist_max_iter)
        # Graph 1 => h-histogram, Graph 2 => h-histograms [FA, BP]

        if not hasattr(self, 'all_distances') or not reuse_distances:
            self.all_distances = np.zeros((len(X), len(X)))
            for i in range(len(list_of_hists)):
                self.all_distances += self.wasserstein_distances(list_of_hists[i], list_of_hists[i], emd_max_iter)

        if self.n_refs >= len(X):
            ref_ids = np.arange(len(X))
        else:
            if self.random_refs == True:
                ref_ids = np.random.choice(len(X), self.n_refs)
            else:
                ref_ids = self.median_cluster_indices(self.all_distances, cluster_max_iter)

        self.ref_ids = ref_ids
        self.ref_list_of_hists = [[hists[i] for i in ref_ids] for hists in list_of_hists]

        return self

    def transform(self, X: List[np.ndarray], bipartite_sets: List[np.ndarray] = [],
                  hist_max_iter: int = 25, emd_max_iter: int = 10000) -> np.ndarray:
        assert hasattr(self, 'ref_list_of_hists'), 'transform should only be called after fit'

        list_of_hists = self.to_separate_hists(X, bipartite_sets, hist_max_iter)

        distances = np.zeros((len(X), self.n_refs))
        for i in range(len(list_of_hists)):
            distances += self.wasserstein_distances(list_of_hists[i], self.ref_list_of_hists[i], emd_max_iter)

        return distances

    def fit_transform(self, X: List[np.ndarray], bipartite_sets: List[np.ndarray] = [],
                      hist_max_iter: int = 25, emd_max_iter: int = 10000, cluster_max_iter: int = 300,
                      reuse_distances: bool = False) -> np.ndarray:
        self.fit(X, bipartite_sets, hist_max_iter, emd_max_iter, cluster_max_iter, reuse_distances)

        return self.all_distances[:, self.ref_ids]

    def fit_subset(self, indices: np.ndarray, cluster_max_iter: int = 300):
        assert hasattr(self, 'all_distances'), 'fit_subset should only be called after fit'

        if self.n_refs >= len(indices):
            ref_ids = indices
        else:
            if self.random_refs:
                ref_ids = np.random.choice(indices, self.n_refs)
            else:
                ref_ids = self.median_cluster_indices(self.all_distances[indices][:, indices], cluster_max_iter)

        self.ref_ids = ref_ids
        # We can't set ref_list_of_hists, but it is not necessary as it does not make sense to call
        # transform instead of transform_subset after fit_subset
        self.ref_list_of_hists = []

        return self

    def transform_subset(self, indices: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'all_distances'), 'transform_subset should only be called after fit'

        return self.all_distances[indices][:, self.ref_ids]

    def fit_transform_subset(self, indices: np.ndarray, cluster_max_iter: int = 300) -> np.ndarray:
        return self.fit_subset(indices, cluster_max_iter).transform_subset(indices)