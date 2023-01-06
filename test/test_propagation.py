from unittest import TestCase
import networkx as nx
import numpy as np

from wwlr import WeisfeilerLehman, ContinuousWeisfeilerLehman


class TestWeisfeilerLehman(TestCase):
    def test_unlabeled(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])

        representation = WeisfeilerLehman(2).fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0, 0, 0],
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

        self.assertTrue([np.testing.assert_array_equal(x, y) for x, y in zip(representation, real_representation)])

    def test_labeled(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        nx.set_node_attributes(graph_1,  {i: {'label': x} for i, x in zip(range(5), [0, 1, 0, 0, 1])})

        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        nx.set_node_attributes(graph_2,  {i: {'label': x} for i, x in zip(range(6), [0, 1, 0, 0, 1, 0])})

        representation = WeisfeilerLehman(3).fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0, 0, 0, 0],
                                         [1, 1, 1, 1],
                                         [0, 2, 2, 2],
                                         [0, 3, 3, 3],
                                         [1, 1, 4, 4]]),
                               np.array([[0,  3,  5,  5],
                                         [1,  4,  6,  6],
                                         [0,  0,  7,  7],
                                         [0,  3,  5,  8],
                                         [1,  4,  8,  9],
                                         [0,  5,  9, 10]])]

        self.assertTrue([np.testing.assert_array_equal(x, y) for x, y in zip(representation, real_representation)])

    def test_different_label(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        nx.set_node_attributes(graph_1,  {i: {'label_2': x} for i, x in zip(range(5), [0, 1, 0, 0, 1])})

        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        nx.set_node_attributes(graph_2,  {i: {'label_2': x} for i, x in zip(range(6), [0, 1, 0, 0, 1, 0])})

        representation = WeisfeilerLehman(3).fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0, 0, 0, 0],
                                         [1, 1, 1, 1],
                                         [0, 2, 2, 2],
                                         [0, 3, 3, 3],
                                         [1, 1, 4, 4]]),
                               np.array([[0,  3,  5,  5],
                                         [1,  4,  6,  6],
                                         [0,  0,  7,  7],
                                         [0,  3,  5,  8],
                                         [1,  4,  8,  9],
                                         [0,  5,  9, 10]])]

        self.assertTrue([np.testing.assert_array_equal(x, y) for x, y in zip(representation, real_representation)])

    def test_directed(self):
        graph_1 = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph_2 = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])

        representation = WeisfeilerLehman(2).fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0, 0, 0],
                                         [1, 1, 1],
                                         [0, 2, 2],
                                         [1, 1, 1],
                                         [1, 3, 3]]),
                               np.array([[1, 4, 4],
                                         [0, 5, 5],
                                         [0, 2, 6],
                                         [1, 1, 7],
                                         [0, 2, 8],
                                         [2, 6, 9]])]

        self.assertTrue([np.testing.assert_array_equal(x, y) for x, y in zip(representation, real_representation)])

    def test_large_graph(self):
        graph = nx.watts_strogatz_graph(100000, 2, 0.05)

        computation_completed = True
        # Either there will be an exception, or the python interpreter will segfault if it cannot handle these graphs
        try:
            WeisfeilerLehman(1).fit_transform([graph])
        except Exception:
            computation_completed = False

        self.assertTrue(computation_completed)

    def test_zero_degree_node(self):
        graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        nx.set_node_attributes(graph,  {i: {'label': x} for i, x in zip(range(5), [0, 1, 0, 0, 1, 2])})
        graph.add_node(5, label=3)

        representation = WeisfeilerLehman(2).fit_transform([graph])
        real_representation = [np.array([[0, 0, 0],
                                         [1, 1, 1],
                                         [0, 2, 2],
                                         [0, 3, 3],
                                         [1, 1, 4],
                                         [2, 4, 5]])]

        self.assertTrue([np.testing.assert_array_equal(x, y) for x, y in zip(representation, real_representation)])

    def test_generator(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])

        graphs = (graph for graph in [graph_1, graph_2])

        representation = WeisfeilerLehman(2).fit_transform(graphs)
        real_representation = WeisfeilerLehman(2).fit_transform([graph_1, graph_2])

        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])


class TestContinuousWeisfeilerLehman(TestCase):
    def test_unlabeled(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])

        representation = ContinuousWeisfeilerLehman(2).fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0.98994949, 0.47140452, 0.32016224],
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

        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])

    def test_labeled(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        nx.set_node_attributes(graph_1, {i: {'label_1': x, 'label_2': y} for i, x, y in
                               zip(range(5), [1.5, 1, 0.1, 3, 1], [170, 250, 1002, 18, 501])})

        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        nx.set_node_attributes(graph_2, {i: {'label_1': x, 'label_2': y} for i, x, y in
                               zip(range(6), [3, 1, 0.5, 2.5, 1, 0.1], [391, 321, 29, 1400, 123, 43])})

        es = ContinuousWeisfeilerLehman(3)
        representation = es.fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0.16271114, -0.5147757,  0.09642141, -0.21315495, -0.00991835,
                                          -0.07057962, -0.06872746,  0.00748239],
                                         [-0.33446178, -0.32427839, -0.43389637,  0.07576598, -0.36760664,
                                          0.14432517, -0.29026863,  0.14049703],
                                         [-1.22937304,  1.46639639, -0.69905526,  0.63892367, -0.4159429,
                                          0.34391741, -0.27243025,  0.21376905],
                                         [1.6542299, -0.87672061,  0.78417729, -0.49870249,  0.40093983,
                                          -0.23160938,  0.19568337, -0.08917047],
                                         [-0.33446178,  0.27340694, -0.06101668,  0.28412242, -0.00922783,
                                          0.1771165, -0.00836468,  0.11663526]]),
                               np.array([[1.6542299,  0.01147313,  0.53559083, -0.24569824,  0.24557329,
                                          -0.21504008,  0.15477018, -0.15759323],
                                         [-0.33446178, -0.15521202, -0.08587532, -0.32189717,  0.00665408,
                                          -0.22188608,  0.02875066, -0.15499705],
                                         [-0.8316347, -0.85052723, -0.00301317, -0.04686667,  0.12128006,
                                          0.0215933,  0.11667661,  0.00279162],
                                         [1.15705698,  2.41412054,  0.28700437,  0.83775525,  0.08399209,
                                          0.38889594,  0.01873815,  0.18212698],
                                         [-0.33446178, -0.62669288, -0.2350272, -0.07306005, -0.21431166,
                                          -0.07087727, -0.17679352, -0.07385379],
                                         [-1.22937304, -0.81719019, -0.78191741, -0.72194154, -0.5084723,
                                          -0.39750079, -0.36139198, -0.23418903]])]

        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])

    def test_directed(self):
        graph_1 = nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph_2 = nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])

        representation = ContinuousWeisfeilerLehman(2).fit_transform([graph_1, graph_2])
        real_representation = [np.array([[0.98994949,  0.98994949,  0.98994949],
                                         [-0.56568542,  0.21213203,  0.60104076],
                                         [0.98994949,  0.60104076,  0.60104076],
                                         [-0.56568542,  0.21213203,  0.60104076],
                                         [-0.56568542, -0.1767767,  0.11490485]]),
                               np.array([[-0.56568542, -0.56568542, -0.56568542],
                                         [0.98994949,  0.21213203, -0.1767767],
                                         [0.98994949,  0.60104076,  0.21213203],
                                         [-0.56568542,  0.21213203,  0.4065864],
                                         [0.98994949,  0.60104076,  0.4065864],
                                         [-2.12132034, -0.56568542,  0.01767767]])]

        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])

    def test_weighted(self):
        graph_1 = nx.Graph([(0, 1, {'weight': 1.0}), (0, 2, {'weight': 0.2}), (0, 3, {'weight': 0.5}),
                            (1, 2, {'weight': 1.0}), (2, 4, {'weight': 0.5}), (3, 4, {'weight': 1.0})])
        nx.set_node_attributes(graph_1,  {i: {'label': x} for i, x in zip(range(5), [1.5, 1, 0.1, 3, 1])})

        graph_2 = nx.Graph([(0, 1, {'weight': 0.5}), (0, 2, {'weight': 1.0}), (1, 2, {'weight': 1.0}),
                            (2, 3, {'weight': 1.0}), (1, 4, {'weight': 0.2}), (3, 4, {'weight': 0.2}),
                            (4, 5, {'weight': 1.0})])
        nx.set_node_attributes(graph_2,  {i: {'label': x} for i, x in zip(range(6), [3, 1, 0.5, 2.5, 1, 0.1])})

        es = ContinuousWeisfeilerLehman(3)
        representation = es.fit_transform([graph_1, graph_2])

        real_representation = [np.array([[0.16271114, -0.16542299, -0.42581731, -0.60566478],
                                         [-0.33446178, -0.43389637, -0.50350057, -0.57376768],
                                         [-1.22937304, -0.98078658, -0.86225227, -0.83181078],
                                         [1.6542299, 0.59773744, 0.09372839, -0.2164609],
                                         [-0.33446178, -0.073446, -0.17598791, -0.33844434]]),
                               np.array([[1.6542299, 0.4112976, -0.0195856, -0.23879052],
                                         [-0.33446178, -0.46704123, -0.51095817, -0.5725984],
                                         [-0.8316347, -0.00301317, 0.00389201, -0.12937795],
                                         [1.15705698, 0.0881352, -0.25739998, -0.43582982],
                                         [-0.33446178, -0.69905526, -0.8468261, -0.92678807],
                                         [-1.22937304, -0.78191741, -0.74048633, -0.79365621]])]

        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])

    def test_large_graph(self):
        graph = nx.watts_strogatz_graph(100000, 2, 0.05)

        computation_completed = True
        # Either there will be an exception, or the python interpreter will segfault if it cannot handle these graphs
        try:
            ContinuousWeisfeilerLehman(1).fit_transform([graph])
        except Exception:
            computation_completed = False

        self.assertTrue(computation_completed)

    def test_zero_degree_node(self):
        graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        nx.set_node_attributes(graph,  {i: {'label': x} for i, x in zip(range(5), [1.5, 1, 0.1, 3, 1])})
        graph.add_node(5, label=3)

        representation = ContinuousWeisfeilerLehman(2).fit_transform([graph])
        real_representation = [np.array([[-0.09325048, -0.15541747, -0.25514368],
                                         [-0.55950288, -0.65275337, -0.59058638],
                                         [-1.39875721, -0.90142131, -0.63591647],
                                         [1.30550673,  0.48956502,  0.13016213],
                                         [-0.55950288, -0.30306406, -0.2544961],
                                         [1.30550673,  1.30550673,  1.30550673]])]
        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])

    def test_generator(self):
        graph_1 = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        graph_2 = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        graphs = (graph for graph in [graph_1, graph_2])

        representation = ContinuousWeisfeilerLehman(2).fit_transform(graphs)
        real_representation = ContinuousWeisfeilerLehman(2).fit_transform([graph_1, graph_2])

        self.assertTrue([np.testing.assert_allclose(x, y, 1e-5) for x, y in zip(representation, real_representation)])
