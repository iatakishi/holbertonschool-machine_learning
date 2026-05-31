#!/usr/bin/env python3
"""Module for creating mini-batches"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


# Create mini-batches for mini-batch gradient descent
def create_mini_batches(X, Y, batch_size):
    """Create mini-batches from shuffled data"""
    X, Y = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []
    for i in range(0, m, batch_size):
        mini_batches.append((X[i:i + batch_size], Y[i:i + batch_size]))
    return mini_batches
