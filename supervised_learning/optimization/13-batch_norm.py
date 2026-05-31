#!/usr/bin/env python3
"""Module for batch normalization"""
import numpy as np


# Normalize an unactivated output using batch normalization
def batch_norm(Z, gamma, beta, epsilon):
    """Normalize an unactivated output of a neural network using batch normalization"""
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
