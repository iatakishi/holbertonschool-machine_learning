#!/usr/bin/env python3
"""
PCA v2 module for dimensionality reduction.
"""
import numpy as np


def pca(X, ndim):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
        ndim (int): the new dimensionality of the transformed X

    Returns:
        numpy.ndarray: T, shape (n, ndim), the transformed version of X
    """
    # Datanı mərkəzləşdiririk (hər sütunun ortalamasını 0-a gətiririk)
    X_m = X - np.mean(X, axis=0)

    # Mərkəzləşdirilmiş X matrisi üzərində SVD tətbiq edirik
    u, s, vh = np.linalg.svd(X_m)

    # İlk `ndim` əsas komponenti (V^T-in sətirlərini)
    # götürüb sütunlara çevirməklə çəki matrisini `W` qururuq
    W = vh[:ndim].T

    # X_m-i yeni fəzaya proyeksiya edərək T-ni əldə edirik
    T = np.matmul(X_m, W)

    return T
