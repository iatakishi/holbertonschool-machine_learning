#!/usr/bin/env python3
"""
PDF module for Gaussian Mixture Model.
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X (numpy.ndarray): shape (n, d) containing the data points whose
            PDF should be evaluated
        m (numpy.ndarray): shape (d,) containing the mean of the
            distribution
        S (numpy.ndarray): shape (d, d) containing the covariance of the
            distribution

    Returns:
        P: numpy.ndarray of shape (n,) containing the PDF values for each
            data point, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    # Kovariasiya matrisinin determinantı və tərsini hesablayırıq
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)

    # Normallaşdırma sabiti: 1 / sqrt((2*pi)^d * det(S))
    norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    # Hər nöqtə üçün mərkəzləşdirilmiş fərq
    X_m = X - m

    # Eksponentin içindəki kvadratik forma: (x-m)^T * S^-1 * (x-m)
    # Bunu vektorlaşdırılmış şəkildə, dövr olmadan hesablayırıq
    exponent = -0.5 * np.sum(np.dot(X_m, inv) * X_m, axis=1)

    P = norm_const * np.exp(exponent)

    # Minimum dəyəri 1e-300 ilə məhdudlaşdırırıq
    P = np.maximum(P, 1e-300)

    return P
