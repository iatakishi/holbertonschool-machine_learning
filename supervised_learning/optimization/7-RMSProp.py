#!/usr/bin/env python3
"""Module for RMSProp optimization"""
import numpy as np


# Update a variable using the RMSProp optimization algorithm
def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable using RMSProp optimization"""
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
