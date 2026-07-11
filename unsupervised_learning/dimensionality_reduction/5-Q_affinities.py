#!/usr/bin/env python3
"""
Q affinities module for t-SNE.
"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities.

    Args:
        Y (numpy.ndarray): shape (n, ndim) containing the low dimensional
            transformation of X
            n is the number of points
            ndim is the new dimensional representation of X

    Returns:
        (Q, num)
            Q: numpy.ndarray of shape (n, n) containing the Q affinities
            num: numpy.ndarray of shape (n, n) containing the numerator of
                the Q affinities
    """
    n, ndim = Y.shape

    # Cütlərarası kvadrat Evklid məsafələrini hesablayırıq:
    # ||yi - yj||^2 = ||yi||^2 - 2*yi.yj + ||yj||^2
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)

    # Student-t (1 sərbəstlik dərəcəli) paylanmaya
    # əsaslanan qeyri-normallaşdırılmış oxşarlıq
    num = 1 / (1 + D)

    # Diaqonalın 0 olduğuna əmin oluruq (özü ilə oxşarlıq nəzərə alınmır)
    np.fill_diagonal(num, 0)

    # Q affinitilərini normallaşdırırıq
    Q = num / np.sum(num)

    return Q, num
