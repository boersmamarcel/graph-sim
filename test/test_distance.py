import numpy as np

from unittest import TestCase
from wwlr import WassersteinDissimilarity


class TestWassersteinDissimilarity(TestCase):
    def test_small_histogram_creation(self):
        samples = np.array([[-0.56568542, 0.21213203, 0.4065864],
                            [0.98994949, 0.73067701, 0.5146166],
                            [0.98994949, 0.47140452, 0.42819244],
                            [-0.56568542, 0.21213203, 0.27695016],
                            [0.98994949, 0.21213203, 0.16891995],
                            [-2.12132034, -0.56568542, -0.1767767]])

        dr = WassersteinDissimilarity(categorical=False, n_refs=2)
        bins, _ = dr.optimal_sparse_histogram(samples)
        self.assertEqual(len(bins), len(samples))

    def test_large_histogram_creation(self):
        samples = np.random.lognormal(size=(100000, 4))

        dr = WassersteinDissimilarity(max_bins=2000, categorical=False, n_refs=2)
        bins, _ = dr.optimal_sparse_histogram(samples)
        self.assertGreater(len(bins), 1000)
        self.assertLess(len(bins), 2000)

    def test_continuous_distance(self):
        embeddings = [np.array([[0.98994949, 0.47140452, 0.32016224],
                                [-0.56568542, 0.21213203, 0.34176828],
                                [0.98994949, 0.47140452, 0.32016224],
                                [-0.56568542, -0.1767767, -0.01473139],
                                [-0.56568542, -0.1767767, -0.01473139]]),
                      np.array([[-0.56568542, 0.21213203, 0.4065864],
                                [0.98994949, 0.73067701, 0.5146166],
                                [0.98994949, 0.47140452, 0.42819244],
                                [-0.56568542, 0.21213203, 0.27695016],
                                [0.98994949, 0.21213203, 0.16891995],
                                [-2.12132034, -0.56568542, -0.1767767]])]

        dr = WassersteinDissimilarity(categorical=False, n_refs=2)
        hists = dr.create_histograms(embeddings)
        distances = dr.wasserstein_distances(hists, hists)
        real_distances = np.array([[0., 0.59982517],
                                   [0.59982517, 0.]])
        np.testing.assert_allclose(distances, real_distances, 1e-3)

    def test_categorical_distance(self):
        embeddings = [np.array([[0, 0, 0],
                                [1, 1, 1],
                                [0, 0, 0],
                                [1, 2, 2],
                                [1, 2, 2]]),
                      np.array([[1, 1, 3],
                                [0, 3, 4],
                                [0, 0, 5],
                                [1, 1, 6],
                                [0, 4, 7],
                                [2, 5, 8]])]

        dr = WassersteinDissimilarity(categorical=True, n_refs=2)
        hists = dr.create_histograms(embeddings)
        distances = dr.wasserstein_distances(hists, hists)
        real_distances = np.array([[0., 0.63333333],
                                   [0.63333333, 0.]])
        np.testing.assert_allclose(distances, real_distances, 1e-3)

    def test_distance_approximation(self):
        embeddings = [np.array([[0, 0, 0],
                                [1, 1, 1],
                                [0, 0, 0],
                                [1, 2, 2],
                                [1, 2, 2]]),
                      np.array([[1, 1, 3],
                                [0, 3, 4],
                                [0, 0, 5],
                                [1, 1, 6],
                                [0, 4, 7],
                                [2, 5, 8]])]

        dr = WassersteinDissimilarity(categorical=True, n_refs=2, approximate=True)
        hists = dr.create_histograms(embeddings)
        distances = dr.wasserstein_distances(hists, hists)

        dr = WassersteinDissimilarity(categorical=True, n_refs=2)
        hists = dr.create_histograms(embeddings)
        real_distances = dr.wasserstein_distances(hists, hists)

        np.testing.assert_allclose(distances, real_distances, 1e-5, 1e-2)

    def test_large_size_distance(self):
        embeddings = [np.random.lognormal(size=(100000, 4)),
                      np.random.normal(size=(100000, 4))]

        computation_completed = True
        try:
            dr = WassersteinDissimilarity(categorical=False, n_refs=2)
            hists = dr.create_histograms(embeddings)
            dr.wasserstein_distances(hists, hists)
        except Exception:
            computation_completed = False

        self.assertTrue(computation_completed)

    def test_clustered(self):
        embeddings = []
        for params in [(0, 1), (0, 5), (5, 1), (5, 5)]:
            for i in range(5):
                embeddings.append(np.random.normal(*params, size=(100, 2)))

        dr = WassersteinDissimilarity(categorical=False, n_refs=4)
        reprs = dr.fit_transform(embeddings)

        # Expect one cluster per distribution
        for i in range(4):
            n_clusters = np.count_nonzero(reprs[i*5:(i+1)*5] == 0)
            self.assertEqual(n_clusters, 1)

        # Distances should be smaller for more similar distributions
        self.assertTrue(np.mean(reprs[5:10, 0]) < np.mean(reprs[10:15, 0]) < np.mean(reprs[15:20, 0]))
        self.assertTrue(np.mean(reprs[0:5, 1]) < np.mean(reprs[15:20, 1]) < np.mean(reprs[10:15, 1]))
        self.assertTrue(np.mean(reprs[15:20, 2]) < np.mean(reprs[0:5, 2]) < np.mean(reprs[5:10, 2]))
        self.assertTrue(np.mean(reprs[10:15, 3]) < np.mean(reprs[5:10, 3]) < np.mean(reprs[0:5, 3]))

    def test_cluster_vs_random(self):
        embeddings = []
        for params in [(0, 1), (0, 5), (5, 1), (5, 5)]:
            for i in range(5):
                embeddings.append(np.random.normal(*params, size=(100, 2)))

        dr = WassersteinDissimilarity(categorical=False, n_refs=4)
        reprs = dr.fit_transform(embeddings)

        dr = WassersteinDissimilarity(categorical=False, n_refs=4, random_refs=True)
        reprs_random = dr.fit_transform(embeddings)

        self.assertEqual(reprs.shape, (20, 4))
        self.assertEqual(reprs_random.shape, (20, 4))

        var_in_distances = np.std(np.mean(reprs, axis=1))
        var_in_distances_random = np.std(np.mean(reprs_random, axis=1))

        # As random references are likely to be less spaced out, the variability in distances to all references
        # should be larger than for well clustered references
        self.assertLess(var_in_distances, var_in_distances_random)

    def test_bipartite(self):
        random_vars = np.concatenate([np.random.normal(size=(100, 2)), np.random.lognormal(size=(100, 2))], axis=0)
        embeddings = [random_vars, random_vars]
        bipartite_sets = [np.arange(0, 200, 2), np.arange(100)]

        dr = WassersteinDissimilarity(n_refs=1, random_refs=True)
        reprs = dr.fit_transform(embeddings, bipartite_sets=bipartite_sets)

        self.assertTrue(reprs[0] == 0 and reprs[1] > 0 or reprs[0] > 0 and reprs[1] == 0)

    def test_fit_transform(self):
        embeddings = []
        for params in [(0, 1), (0, 5), (5, 1), (5, 5)]:
            for i in range(5):
                embeddings.append(np.random.normal(*params, size=(100, 2)))

        dr = WassersteinDissimilarity(n_refs=4, categorical=False)
        reprs_1 = dr.fit_transform(embeddings)

        dr = WassersteinDissimilarity(n_refs=4, categorical=False)
        reprs_2 = dr.fit(embeddings).transform(embeddings)

        np.testing.assert_allclose(reprs_1, reprs_2, atol=1e-07)

    def test_subset(self):
        embeddings = []
        for params in [(0, 1), (0, 5), (5, 1), (5, 5)]:
            for i in range(5):
                embeddings.append(np.random.normal(*params, size=(100, 2)))

        dr = WassersteinDissimilarity(n_refs=2, categorical=False)
        dr.fit(embeddings)
        dr.fit_subset(np.arange(10))
        reprs_1 = dr.transform_subset(np.arange(10, 20))

        dr = WassersteinDissimilarity(n_refs=2, categorical=False)
        dr.fit(embeddings[:10])
        reprs_2 = dr.transform(embeddings[10:])

        np.testing.assert_allclose(reprs_1, reprs_2)

    def test_reuse_distances(self):
        embeddings = []
        for params in [(0, 1), (0, 5), (5, 1), (5, 5)]:
            for i in range(5):
                embeddings.append(np.random.normal(*params, size=(100, 2)))

        dr = WassersteinDissimilarity(n_refs=4, categorical=False)
        reprs = dr.fit_transform(embeddings)
        self.assertEqual(reprs.shape, (20, 4))
        all_distances = dr.all_distances

        dr.n_refs = 2
        reprs = dr.fit_transform(embeddings, reuse_distances=True)
        self.assertEqual(reprs.shape, (20, 2))
        np.testing.assert_allclose(all_distances, dr.all_distances)

        z = np.random.uniform(size=(20, 1))
        distances = np.sum((z[:, np.newaxis] - z[np.newaxis, :]) ** 2, axis=-1)

        dr.all_distances = distances
        reprs = dr.fit_transform(embeddings, reuse_distances=True)
        ids = dr.ref_ids
        np.testing.assert_array_equal(reprs, distances[:, ids])
