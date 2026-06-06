#!/usr/bin/env python3
"""Module for forward propagation with dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Arguments:
    X -- numpy.ndarray of shape (nx, m) containing the input data
    weights -- dictionary of the weights and biases of the neural network
    L -- number of layers in the network
    keep_prob -- probability that a node will be kept

    Returns:
    cache -- dictionary containing outputs and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.dot(W, A_prev) + b

        if i == L:
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = A * D
            A = A / keep_prob
            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache
