#!/usr/bin/env python3
"""
Entropy module for t-SNE.
"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data
    point.

    Args:
        Di (numpy.ndarray): shape (n - 1,) containing the pairwise
            distances between a data point and all other points except
            itself
            n is the number of data points
        beta (numpy.ndarray): shape (1,) containing the beta value for the
            Gaussian distribution

    Returns:
        (Hi, Pi)
            Hi: the Shannon entropy of the points
            Pi: numpy.ndarray of shape (n - 1,) containing the P affinities
                of the points
    """
    # Qeyri-normallaşdırılmış Qauss oxşarlıqları:
    # exp(-Di * beta)
    Pi = np.exp(-Di * beta)

    # Normallaşdırma sabiti
    sum_Pi = np.sum(Pi)

    # P affinitilərini normallaşdırırıq
    Pi = Pi / sum_Pi

    # Şennon entropiyası: H = -sum(Pi * log2(Pi))
    # Amma daha stabil formula istifadə edirik (bax aşağıda izah)
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
