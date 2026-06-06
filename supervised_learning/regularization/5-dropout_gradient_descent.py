#!/usr/bin/env python3
"""Module for gradient descent with dropout."""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural
    network with Dropout using gradient descent.

    Arguments:
    Y -- one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights -- dictionary of the weights and biases of the neural network
    cache -- dictionary of outputs and dropout masks of each layer
    alpha -- learning rate
    keep_prob -- probability that a node will be kept
    L -- number of layers of the network
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.dot(W.T, dZ)
            D = cache['D' + str(i - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - np.power(cache['A' + str(i - 1)], 2))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
