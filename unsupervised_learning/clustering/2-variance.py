#!/usr/bin/env python3
"""
Variance module for K-means.
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        C (numpy.ndarray): shape (k, d) containing the centroid means for
            each cluster

    Returns:
        var: the total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    try:
        # Hər data nöqtəsi ilə hər centroid arasındakı
        # kvadrat Evklid məsafələrini vektorlaşdırılmış şəkildə hesablayırıq
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Hər nöqtə üçün ən yaxın centroidə olan minimum məsafəni tapırıq
        min_distances = np.min(distances, axis=1)

        # Bu minimum məsafələrin kvadratlarının cəmi ümumi dispersiyanı verir
        var = np.sum(min_distances ** 2)

        return var
    except Exception:
        return None
