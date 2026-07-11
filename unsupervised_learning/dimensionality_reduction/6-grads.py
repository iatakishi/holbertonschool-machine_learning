#!/usr/bin/env python3
"""
Gradients module for t-SNE.
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): shape (n, ndim) containing the low dimensional
            transformation of X
        P (numpy.ndarray): shape (n, n) containing the P affinities of X

    Returns:
        (dY, Q)
            dY: numpy.ndarray of shape (n, ndim) containing the gradients
                of Y
            Q: numpy.ndarray of shape (n, n) containing the Q affinities
                of Y
    """
    n, ndim = Y.shape

    Q, num = Q_affinities(Y)

    # (P - Q) fərqi hər cüt nöqtə üçün "cəlbetmə/itələmə" qüvvəsini verir
    PQ_diff = P - Q

    dY = np.zeros((n, ndim))

    for i in range(n):
        # Hər i nöqtəsi üçün qradiyenti bütün digər
        # nöqtələr üzərindən cəmləyirik:
        # dY_i = sum_j (P_ij - Q_ij) * (Y_i - Y_j) * num_ij
        diff = (PQ_diff[:, i] * num[:, i]).reshape(-1, 1) * (Y[i, :] - Y)
        dY[i, :] = np.sum(diff, axis=0)

    return dY, Q
