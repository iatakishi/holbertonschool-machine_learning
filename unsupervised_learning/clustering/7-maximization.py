#!/usr/bin/env python3
"""
Maximization module for EM algorithm (GMM).
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        g (numpy.ndarray): shape (k, n) containing the posterior
            probabilities for each data point in each cluster

    Returns:
        (pi, m, S)
            pi: numpy.ndarray of shape (k,) containing the updated priors
                for each cluster
            m: numpy.ndarray of shape (k, d) containing the updated
                centroid means for each cluster
            S: numpy.ndarray of shape (k, d, d) containing the updated
                covariance matrices for each cluster
        or (None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    # Hər nöqtənin klasterlər üzrə cəmi 1 olmalıdır (posterior ehtimallar)
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # Hər klasterin "effektiv" nöqtə sayı
    Nk = np.sum(g, axis=1)

    # Yenilənmiş priors
    pi = Nk / n

    # Yenilənmiş centroid ortalamaları: (k, d)
    m = np.matmul(g, X) / Nk[:, np.newaxis]

    S = np.zeros((k, d, d))

    for j in range(k):
        # Hər klaster üçün mərkəzləşdirilmiş fərq
        X_m = X - m[j]

        # Kovariasiya matrisi: sum_i g[j,i] * (x_i - m_j)(x_i - m_j)^T / Nk[j]
        S[j] = np.matmul(g[j] * X_m.T, X_m) / Nk[j]

    return pi, m, S
