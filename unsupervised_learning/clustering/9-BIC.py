#!/usr/bin/env python3
"""
BIC module for GMM.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data set
        kmin (int): positive integer containing the minimum number of
            clusters to check for (inclusive)
        kmax (int): positive integer containing the maximum number of
            clusters to check for (inclusive). If None, set to the maximum
            number of clusters possible
        iterations (int): positive integer containing the maximum number
            of iterations for the EM algorithm
        tol (float): non-negative float containing the tolerance for the
            EM algorithm
        verbose (bool): determines if the EM algorithm should print
            information to standard output

    Returns:
        (best_k, best_result, l, b)
            best_k: the best value for k based on its BIC
            best_result: tuple containing (pi, m, S) for the best k
            l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
                likelihood for each cluster size tested
            b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
                value for each cluster size tested
        or (None, None, None, None) on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    log_likelihoods = []
    bic_values = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or g is None or \
                log_likelihood is None:
            return None, None, None, None

        # Modelin parametr sayı:
        # priors (k-1, çünki cəmi 1-ə bərabərdir) +
        # centroid ortalamaları (k * d) +
        # kovariasiya matrisləri (k * d * (d+1) / 2, simmetrik olduğu üçün)
        p = (k - 1) + k * d + k * d * (d + 1) / 2

        bic = p * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        log_likelihoods.append(log_likelihood)
        bic_values.append(bic)

    log_likelihoods = np.array(log_likelihoods)
    bic_values = np.array(bic_values)

    best_index = np.argmin(bic_values)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, log_likelihoods, bic_values
