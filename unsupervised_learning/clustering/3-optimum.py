#!/usr/bin/env python3
"""
Optimize k module for K-means.
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        kmin (int): positive integer containing the minimum number of
            clusters to check for (inclusive)
        kmax (int): positive integer containing the maximum number of
            clusters to check for (inclusive)
        iterations (int): positive integer containing the maximum number
            of iterations for K-means

    Returns:
        (results, d_vars)
            results: list containing the outputs of K-means for each
                cluster size
            d_vars: list containing the difference in variance from the
                smallest cluster size for each cluster size
        or (None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Ən azı 2 fərqli klaster ölçüsü analiz olunmalıdır
    if kmax - kmin < 1:
        return None, None

    try:
        results = []
        variances = []

        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k, iterations)
            if C is None or clss is None:
                return None, None

            results.append((C, clss))
            variances.append(variance(X, C))

        # Ən kiçik klaster ölçüsünün (kmin) dispersiyasından fərqi hesablayırıq
        base_var = variances[0]
        d_vars = [base_var - v for v in variances]

        return results, d_vars
    except Exception:
        return None, None
