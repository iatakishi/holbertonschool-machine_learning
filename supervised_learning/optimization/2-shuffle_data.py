#!/usr/bin/env python3
"""Module for shuffling data matrices"""
import numpy as np


# Shuffle two matrices in the same order
def shuffle_data(X, Y):
    """Shuffle the data points in two matrices the same way"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
