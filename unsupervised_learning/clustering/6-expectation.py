#!/usr/bin/env python3
"""
Expectation module for EM algorithm (GMM).
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        pi (numpy.ndarray): shape (k,) containing the priors for each
            cluster
        m (numpy.ndarray): shape (k, d) containing the centroid means for
            each cluster
        S (numpy.ndarray): shape (k, d, d) containing the covariance
            matrices for each cluster

    Returns:
        (g, l)
            g: numpy.ndarray of shape (k, n) containing the posterior
                probabilities for each data point in each cluster
            l: the total log likelihood
        or (None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Hər klaster üçün pi_j * P(x | cluster_j) dəyərlərini toplayacağımız
    # (k, n) formalı matris
    P = np.zeros((k, n))

    for j in range(k):
        P[j] = pi[j] * pdf(X, m[j], S[j])

    # Hər data nöqtəsi üçün klasterlər üzrə cəm (marjinal ehtimal)
    total = np.sum(P, axis=0)

    # Posterior ehtimalları (məsuliyyətlər) normallaşdırırıq
    g = P / total

    # Ümumi log-likelihood
    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
