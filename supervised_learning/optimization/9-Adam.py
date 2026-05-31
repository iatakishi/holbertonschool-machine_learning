#!/usr/bin/env python3
"""Module for Adam optimization"""
import numpy as np


# Update a variable using the Adam optimization algorithm
def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable using Adam optimization"""
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
