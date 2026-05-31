#!/usr/bin/env python3
"""Module for calculating normalization constants"""
import numpy as np


# Calculate normalization constants for a matrix
def normalization_constants(X):
    """Calculate the mean and standard deviation of each feature"""
    return X.mean(axis=0), X.std(axis=0)
