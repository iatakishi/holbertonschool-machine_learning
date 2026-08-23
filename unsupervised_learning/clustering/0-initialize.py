#!/usr/bin/env python3
"""
Initialize K-means module.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset that will be
            used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        k (int): positive integer containing the number of clusters

    Returns:
        numpy.ndarray: shape (k, d) containing the initialized centroids
            for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    # Hər ölçü (dimension) üzrə minimum və maksimum dəyərləri tapırıq
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # Multivariate uniform paylanma ilə (k, d) formalı
    # centroidləri bir dəfəyə yaradırıq
    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    return centroids
