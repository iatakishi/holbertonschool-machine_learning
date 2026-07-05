#!/usr/bin/env python3
"""
PCA module for dimensionality reduction.
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var (float): the fraction of the variance that the PCA transformation
            should maintain.

    Returns:
        numpy.ndarray: the weights matrix, W, of shape (d, nd) that maintains
        var fraction of X's original variance, where nd is the new
        dimensionality of the transformed X.
    """
    # X matrisi üzərində SVD tətbiq edirik
    u, s, vh = np.linalg.svd(X)

    # Hər bir komponentin datasetin nə qədər
    # dispersiyasını izah etdiyini tapırıq
    # Sinqulyar dəyərlərin (s) kvadratı həmin
    # komponentin dispersiyasına bərabərdir
    cum_var = np.cumsum(s ** 2) / np.sum(s ** 2)

    # Kümülatif dispersiya verilmiş
    # `var` həddinə çatana (və ya keçənə) qədər
    # olan minimum komponent sayını (nd) tapırıq
    nd = np.argwhere(cum_var >= var)[0, 0] + 1

    # Çəki matrisi `W` əsas komponentləri
    # (V^T -in sətirləri) götürüb sütunlara (T) çevirməklə formalaşır.
    W = vh[:nd].T

    return W
