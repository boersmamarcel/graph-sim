import numpy as np
import os

from tensorflow.train import Checkpoint
from tensorflow import Variable
from typing import List, Tuple, Union
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorboard.plugins import projector
from tqdm import tqdm


class LaplacianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lap=1e-1):
        self.lap = lap

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return np.exp(-self.lap * X)


def read_labels(filename: str) -> List[str]:
    '''
    Reads labels from a file. Labels are supposed to be stored in each
    line of the file. No further pre-processing will be performed.
    '''
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels


def file_sort_criteria(filename: str) -> Union[str, int]:
    try:
        return int(os.path.splitext(filename)[0])
    except Exception:
        return filename


def retrieve_graph_filenames(data_directory: str) -> List[str]:
    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('gexf')]
    graphs.sort(key=file_sort_criteria)
    return [os.path.join(data_directory, g) for g in graphs]


def retrieve_binary_filenames(data_directory: str) -> List[str]:
    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('pickle')]
    graphs.sort(key=file_sort_criteria)
    return [os.path.join(data_directory, g) for g in graphs]


def sparse_histogram(X: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(np.min(X), np.max(X), n_bins+1)[:-1]  # to not include the max value
    indices = np.digitize(X, bins)

    unique_bins, counts = np.unique(indices, return_counts=True, axis=0)  # type:ignore
    middlepoint_bins = bins[unique_bins - 1] + (bins[1] - bins[0]) / 2
    return middlepoint_bins, counts  # type:ignore


def cv_classify(prepared_model, y: np.ndarray, folds: int = 10,
                n_jobs: int = 1) -> Tuple[List[float], List[float]]:
    param_grid = ParameterGrid({'svc__C': np.logspace(-2, 3, num=6),
                                'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']})
    skf = StratifiedKFold(n_splits=folds)

    def calculate_scores(params):
        means = []
        stds = []

        for col in y.T:
            vals, counts = np.unique(col, return_counts=True)
            class_with_enough = vals[np.argwhere(counts >= 20)]
            indices = np.isin(col, class_with_enough)

            scores = []
            for train_i, test_i in skf.split(np.zeros(np.sum(indices)), col[indices]):
                i = np.arange(len(col))[indices]
                x_train = prepared_model.fit_transform_subset(i[train_i])
                x_test = prepared_model.transform_subset(i[test_i])
                y_train = col[indices][train_i]
                y_test = col[indices][test_i]

                pipe = Pipeline([('scaler', StandardScaler()), ('svc', svm.SVC(tol=1e-2, max_iter=20000000))])
                pipe.set_params(**params)
                pipe.fit(x_train, y_train)
                y_pred = pipe.predict(x_test)
                scores.append(accuracy_score(y_test, y_pred))

            means.append(np.mean(scores))
            stds.append(np.std(scores))

        return means, stds

    if n_jobs > 1:
        try:
            # Uses pathos in the background
            from p_tqdm import p_map
            results = p_map(calculate_scores, param_grid, num_cpus=n_jobs)
        except Exception:
            from pathos.pools import ProcessPool
            p = ProcessPool(n_jobs)
            results = p.map(calculate_scores, param_grid)
    else:
        results = list(map(calculate_scores, tqdm(param_grid)))

    all_means, all_stds = [np.array(t) for t in zip(*results)]

    best_iterations = np.argmax(all_means, axis=0)
    best_means = np.diag(all_means[best_iterations])
    best_stds = np.diag(all_stds[best_iterations])
    return best_means, best_stds


def cv_classify_kernel(prepared_model, y: np.ndarray, folds: int = 10,
                       n_jobs: int = 1) -> Tuple[List[float], List[float]]:
    param_grid = ParameterGrid({'svc__C': np.logspace(-2, 3, num=6),
                                'laplacian__lap': np.logspace(-4, 1, num=6)})
    skf = StratifiedKFold(n_splits=folds)

    def calculate_scores(params):
        means = []
        stds = []

        for col in y.T:
            vals, counts = np.unique(col, return_counts=True)
            class_with_enough = vals[np.argwhere(counts >= 20)]
            indices = np.isin(col, class_with_enough)

            scores = []
            for train_i, test_i in skf.split(np.zeros(np.sum(indices)), col[indices]):
                i = np.arange(len(col))[indices]
                x_train = prepared_model.fit_transform_subset(i[train_i])
                x_test = prepared_model.transform_subset(i[test_i])
                y_train = col[indices][train_i]
                y_test = col[indices][test_i]

                pipe = Pipeline([('laplacian', LaplacianTransformer()),
                                 ('svc', svm.SVC(tol=1e-2, max_iter=20000000, kernel='precomputed'))])
                pipe.set_params(**params)
                pipe.fit(x_train, y_train)
                y_pred = pipe.predict(x_test)
                scores.append(accuracy_score(y_test, y_pred))

            means.append(np.mean(scores))
            stds.append(np.std(scores))

        return means, stds

    if n_jobs > 1:
        try:
            # Uses pathos in the background
            from p_tqdm import p_map
            results = p_map(calculate_scores, param_grid, num_cpus=n_jobs)
        except Exception:
            from pathos.pools import ProcessPool
            p = ProcessPool(n_jobs)
            results = p.map(calculate_scores, param_grid)
    else:
        results = list(map(calculate_scores, tqdm(param_grid)))

    all_means, all_stds = [np.array(t) for t in zip(*results)]

    best_iterations = np.argmax(all_means, axis=0)
    best_means = np.diag(all_means[best_iterations])
    best_stds = np.diag(all_stds[best_iterations])
    return best_means, best_stds



def cv_classify_automl(prepared_model, y: np.ndarray, folds: int = 10) -> Tuple[List[float], List[float]]:
    from autosklearn.experimental.askl2 import AutoSklearn2Classifier

    skf = StratifiedKFold(n_splits=folds)

    means = []
    stds = []

    for col in y.T:
        vals, counts = np.unique(col, return_counts=True)
        class_with_enough = vals[np.argwhere(counts >= 20)]
        indices = np.isin(col, class_with_enough)

        scores = []
        for train_i, test_i in skf.split(np.zeros(np.sum(indices)), col[indices]):
            i = np.arange(len(col))[indices]
            x_train = prepared_model.fit_transform_subset(i[train_i])
            x_test = prepared_model.transform_subset(i[test_i])
            y_train = col[indices][train_i]
            y_test = col[indices][test_i]

            clf = AutoSklearn2Classifier(
                time_left_for_this_task=60*30
            )
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            scores.append(accuracy_score(y_test, y_pred))

        means.append(np.mean(scores))
        stds.append(np.std(scores))

    return means, stds


def create_projection(X: np.ndarray, path: str) -> None:
    ckpt = Checkpoint(embedding=Variable(X))
    ckpt.save(os.path.join(path, "embedding.ckpt"))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(path, config)


def create_metadata_file(graph_filenames, y, path):
    with open(os.path.join(path, 'metadata.tsv'), "w") as f:
        f.write('network')
        for col in y.columns:
            f.write(f'\t{col}')
        f.write('\n')

        for e, name in enumerate(graph_filenames):
            f.write(f'{name}')
            for label in y.iloc[e]:
                f.write(f'\t{label}')
            f.write('\n')
