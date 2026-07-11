#!/usr/bin/env python3
"""
Initialize GMM module.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        k (int): positive integer containing the number of clusters

    Returns:
        (pi, m, S)
            pi: numpy.ndarray of shape (k,) containing the priors for each
                cluster, initialized evenly
            m: numpy.ndarray of shape (k, d) containing the centroid means
                for each cluster, initialized with K-means
            S: numpy.ndarray of shape (k, d, d) containing the covariance
                matrices for each cluster, initialized as identity matrices
        or (None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    # Hər klaster üçün prior ehtimalları bərabər paylayırıq
    pi = np.full((k,), 1 / k)

    # Centroid ortalamalarını K-means ilə initialize edirik
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # Kovariasiya matrislərini eyni (identity) matrislər kimi initialize
    # edirik: (k, d, d) formalı tenzor üçün np.eye(d)-i k dəfə təkrarlayırıq
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
