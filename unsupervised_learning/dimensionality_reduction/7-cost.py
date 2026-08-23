#!/usr/bin/env python3
"""
Cost module for t-SNE.
"""
import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation.

    Args:
        P (numpy.ndarray): shape (n, n) containing the P affinities
        Q (numpy.ndarray): shape (n, n) containing the Q affinities

    Returns:
        C: the cost of the transformation
    """
    # 0-a bölünmə xətalarının qarşısını almaq üçün
    # P və Q-nun ən kiçik dəyərlərini 1e-12 ilə məhdudlaşdırırıq
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    # Kullback-Leibler divergensiyası:
    # C = sum_i sum_j P_ij * log(P_ij / Q_ij)
    C = np.sum(P * np.log(P / Q))

    return C
