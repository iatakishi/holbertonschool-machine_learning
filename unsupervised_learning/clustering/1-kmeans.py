#!/usr/bin/env python3
"""
K-means module.
"""
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions for each data point
        k (int): positive integer containing the number of clusters
        iterations (int): positive integer containing the maximum number of
            iterations that should be performed

    Returns:
        (C, clss)
            C: numpy.ndarray of shape (k, d) containing the centroid means
                for each cluster
            clss: numpy.ndarray of shape (n,) containing the index of the
                cluster in C that each data point belongs to
        or (None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Centroidləri multivariate uniform paylanma ilə initialize edirik
    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    for i in range(iterations):
        C_prev = C.copy()

        # Hər data nöqtəsi ilə hər centroid arasındakı məsafələri
        # vektorlaşdırılmış şəkildə hesablayırıq
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Hər nöqtəni ən yaxın centroidə təyin edirik
        clss = np.argmin(distances, axis=1)

        # Hər klaster üçün yeni centroidi (ortalamanı) hesablayırıq
        for j in range(k):
            if np.sum(clss == j) == 0:
                # Boş klaster varsa, centroidini yenidən initialize edirik
                C[j] = np.random.uniform(min_vals, max_vals, size=(1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        # Əgər centroidlərdə heç bir dəyişiklik baş verməyibsə,
        # erkən dayanırıq
        if np.array_equal(C, C_prev):
            distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
            clss = np.argmin(distances, axis=1)
            return C, clss

    # Son iterasiyadan sonra klaster təyinatlarını yenidən hesablayırıq
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
