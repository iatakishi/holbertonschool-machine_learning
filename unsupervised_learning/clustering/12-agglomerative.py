#!/usr/bin/env python3
"""
Agglomerative clustering module.
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        dist (float): the maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) containing the cluster indices
            for each data point
    """
    # Ward linkage üsulu ilə iyerarxik klasterləşdirmə aparırıq
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Dendroqramı göstəririk, hər klaster fərqli rəngdə olacaq şəkildə
    # (color_threshold verilmiş max cophenetic məsafəyə uyğun olaraq
    # rəngləri ayırır)
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    # Verilmiş maksimum cophenetic məsafəyə əsasən klaster indekslərini
    # təyin edirik
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist,
                                             criterion='distance')

    return clss
