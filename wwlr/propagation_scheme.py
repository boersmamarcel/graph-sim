import copy
import networkx as nx
import numpy as np
import scipy.sparse as ss

from tqdm.auto import tqdm
from collections import defaultdict
from typing import List, Tuple, Optional, Iterable
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin


class WeisfeilerLehman(TransformerMixin):
    """
    Class that implements the Weisfeiler-Lehman transform
    Credits: Christian Bock and Bastian Rieck
    """

    # TODO remove unused state variables
    def __init__(self, num_iterations: int = 3) -> None:
        self._relabel_steps = defaultdict(dict)
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self.num_iterations = num_iterations

    def _reset_label_generation(self) -> None:
        self._last_new_label = -1
        self._label_dict = {}

    def _get_next_label(self) -> int:
        self._last_new_label += 1
        return self._last_new_label

    def _relabel_graphs(self, X: List[nx.Graph]) -> List[nx.Graph]:
        preprocessed_graphs = []
        for i, g in enumerate(X):
            x = g.copy()

            feature_names = [k for k in next(iter(x.nodes().values())).keys() if k != 'bipartite']

            if len(feature_names) == 0:
                label_dict = nx.degree(x)
            elif feature_names[0] != 'label':
                label_dict = x.nodes(data=feature_names[0])
            else:
                label_dict = {}

            if label_dict:
                nx.set_node_attributes(x, {k: {'label': v} for k, v in label_dict})          

            for node, label in nx.get_node_attributes(x, 'label').items():
                if label not in self._preprocess_relabel_dict.keys():
                    self._preprocess_relabel_dict[label] = self._get_next_label()

                x.nodes()[node]['label'] = (self._preprocess_relabel_dict[label])

            self._label_sequences[i][:, 0] = list(nx.get_node_attributes(x, 'label').values())
            preprocessed_graphs.append(x)
        self._reset_label_generation()
        return preprocessed_graphs

    def fit_transform(self, X: Iterable[nx.Graph]) -> List[np.ndarray]:
        X = list(X)
        self._label_sequences = [
            np.empty((g.number_of_nodes(), self.num_iterations+1), dtype=int) for g in X
        ]
        X = self._relabel_graphs(X)
        for it in np.arange(1, self.num_iterations+1, 1):
            self._reset_label_generation()

            for i, g in enumerate(X):
                # Get labels of current interation
                current_labels = list(nx.get_node_attributes(g, 'label').values())

                # Get for each vertex the labels of its neighbors
                neighbor_labels = self._get_neighbor_labels(g, sort=True)

                # Prepend the vertex label to the list of labels of its neighbors
                merged_labels = [[b]+a for a, b in zip(neighbor_labels, current_labels)]

                # Generate a label dictionary based on the merged labels
                self._append_label_dict(merged_labels)

                # Relabel the graph
                new_labels = self._relabel_graph(g, merged_labels)
                self._relabel_steps[i][it] = {idx: {old_label: new_labels[idx]}
                                              for idx, old_label in enumerate(current_labels)}
                nx.set_node_attributes(g, {k: {'label': v} for k, v in zip(range(len(new_labels)), new_labels)})

                self._label_sequences[i][:, it] = new_labels
        return self._label_sequences

    def _relabel_graph(self, X: nx.Graph, merged_labels: List[List[int]]) -> List[int]:
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str, merged))])
        return new_labels

    def _append_label_dict(self, merged_labels: List[List[int]]) -> None:
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str, merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[dict_key] = self._get_next_label()

    def _get_neighbor_labels(self, X: nx.Graph, sort: bool = True) -> List[List[int]]:
        if isinstance(X, nx.DiGraph):
            neighbor_indices = [[n for n in X.predecessors(x)] for x in X.nodes()]
        else:
            neighbor_indices = [[n for n in X.neighbors(x)] for x in X.nodes()]

        neighbor_labels = []
        for n_indices in neighbor_indices:
            labels = [X.nodes('label')[n] for n in n_indices]
            if sort:
                neighbor_labels.append(sorted(labels))
            else:
                neighbor_labels.append(labels)
        return neighbor_labels


class ContinuousWeisfeilerLehman(TransformerMixin):
    """
    Class that implements the continuous Weisfeiler-Lehman propagation scheme
    """

    # TODO make scaling optional
    def __init__(self, num_iterations: int = 3) -> None:
        self.scaler = StandardScaler()
        self.num_iterations = num_iterations

    def _preprocess_graph(self, graph: nx.Graph) -> Tuple[np.ndarray, ss.csc_matrix]:
        node_features = np.array([[v for k,v in d.items() if k != 'bipartite'] for _, d in graph.nodes(data=True)], dtype=float)
        if node_features.shape[1] == 0:
            node_features = np.array([v for _, v in nx.degree(graph)], dtype=float).reshape(-1, 1)

        self.n_features = node_features.shape[1]

        adj_mat = nx.adjacency_matrix(graph).transpose()

        return node_features, adj_mat  # type:ignore

    def fit_transform(self, X: Iterable[nx.Graph]) -> List[np.ndarray]:
        label_sequences = []
        for graph in tqdm(X):
            node_features, adj_mat = self._preprocess_graph(graph)

            self.scaler.partial_fit(node_features)

            avg_deg = adj_mat.getnnz(axis=1).reshape(-1, 1)
            zeros = np.nonzero(avg_deg == 0)[0]
            non_zeros = np.nonzero(avg_deg != 0)[0]

            graph_feat = np.empty((adj_mat.shape[0], self.num_iterations+1, node_features.shape[1]))
            graph_feat[:, 0] = node_features

            for it in range(1, self.num_iterations+1):
                # Some nodes might have degree zero, which would break the division and half their value each
                # iteration. We could remedy this by altering values in the adjacency matrix, but this is very
                # slow due to the csr sparse format.
                agg_feat = adj_mat.dot(graph_feat[:, it-1])[non_zeros] / avg_deg[non_zeros]
                graph_feat[non_zeros, it] = (agg_feat + graph_feat[non_zeros, it-1]) / 2
                graph_feat[zeros, it] = graph_feat[zeros, it-1]

            label_sequences.append(graph_feat)

        def scale_per_feature(x):
            return self.scaler.transform(x.reshape(-1, self.n_features)).reshape(-1, self.n_features*(self.num_iterations+1))
        label_sequences = list(map(scale_per_feature, label_sequences))

        return label_sequences
