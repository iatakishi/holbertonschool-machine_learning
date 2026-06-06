#!/usr/bin/env python3
"""Module for calculating L2 regularization cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of weights and biases of the neural network
        L: number of layers in the neural network
        m: number of data points used

    Returns:
        cost of the network accounting for L2 regularization
    """
    frobenius_sum = sum(
        np.linalg.norm(weights['W' + str(i)]) ** 2
        for i in range(1, L + 1)
    )
    l2_cost = cost + (lambtha / (2 * m)) * frobenius_sum
    return l2_cost
