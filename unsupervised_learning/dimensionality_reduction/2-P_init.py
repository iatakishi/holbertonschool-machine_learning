#!/usr/bin/env python3
"""
Initialize t-SNE module.
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in
    t-SNE.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset to be
            transformed by t-SNE
            n is the number of data points
            d is the number of dimensions in each point
        perplexity (float): the perplexity that all Gaussian distributions
            should have

    Returns:
        (D, P, betas, H)
            D: numpy.ndarray of shape (n, n) with the squared pairwise
                distance between two data points, diagonal is 0
            P: numpy.ndarray of shape (n, n) initialized to all 0's
            betas: numpy.ndarray of shape (n, 1) initialized to all 1's
            H: the Shannon entropy for perplexity, with a base of 2
    """
    n, d = X.shape

    # Cütlərarası kvadrat Evklid məsafələrini hesablayırıq:
    # ||xi - xj||^2 = ||xi||^2 - 2*xi.xj + ||xj||^2
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Diaqonalın tam olaraq 0 olduğuna əmin oluruq
    np.fill_diagonal(D, 0)

    # P affinitiləri üçün sıfırlarla dolu matris
    P = np.zeros((n, n))

    # Beta dəyərləri (1 / (2 * sigma^2)) üçün vahidlərlə dolu matris
    betas = np.ones((n, 1))

    # Verilmiş perplexity üçün Şennon entropiyası (əsas 2)
    H = np.log2(perplexity)

    return D, P, betas, H
