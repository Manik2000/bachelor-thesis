import numpy as np
import itertools
from sklearn.metrics import pairwise_distances, adjusted_rand_score, accuracy_score, confusion_matrix
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment


def connectivity(data, labels, neighbors=10, metric="euclidean"):
    """Compute the connectivity index for given dataset and clustering labels."""
    distances = pairwise_distances(data, metric=metric)
    nearest = distances.argsort(axis=0)[1:(neighbors + 1), :]
    nrow, ncol = nearest.shape
    replicated_indexes = np.tile(labels, (nrow, 1))
    sorted_labels = labels[nearest]
    arr = (replicated_indexes != sorted_labels)
    return  (arr * np.tile(np.array(1 / np.arange(1, neighbors + 1)), (ncol, 1)).T).sum()


def jaccard(labels1, labels2):
    """Compute the Jaccard similarity between two sets of clustering labels."""
    n11 = n10 = n01 = 0
    n = len(labels1)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)


def cluster_stability(X, est, n_iter=20, random_seed=50, **params):
    """Compute the clustering stability index for given dataset and clustering algorithm."""
    np.random.seed(random_seed)
    initial_clusterer = est(**params)
    initial_cluster = initial_clusterer.fit(X).labels_
    nrow = X.shape[0]
    indices = np.arange(nrow)
    scores = []
    for _ in range(n_iter):
        est_ = est(**params)
        sample_indices = np.random.randint(0, nrow, nrow)
        X_bootstrap = X[sample_indices]
        bootstrap_labels = est_.fit(X_bootstrap).labels_
        relabel = -np.ones(nrow)
        relabel[sample_indices] = bootstrap_labels
        in_both = np.intersect1d(indices, sample_indices)
        scores.append(jaccard(initial_cluster[in_both], relabel[in_both]))
    return scores


def _make_cost_m(cm):
    """Define the cost matrix used for optimal assignment problem."""
    s = np.max(cm)
    return (- cm + s)


def clustering_agreement(true_row_labels, predicted_row_labels):
    """Compute the clustering agreement index for real and clustering labels."""
    cm = confusion_matrix(predicted_row_labels, true_row_labels)
    indexes = linear_sum_assignment(_make_cost_m(cm))
    row, column = indexes
    return cm[row, column].sum() / np.sum(cm)
