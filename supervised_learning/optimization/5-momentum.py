#!/usr/bin/env python3
"""Module for gradient descent with momentum"""
import numpy as np


# Update a variable using gradient descent with momentum
def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using gradient descent with momentum"""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
