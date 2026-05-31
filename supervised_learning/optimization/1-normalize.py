#!/usr/bin/env python3
"""Module for normalizing a matrix"""
import numpy as np


# Normalize a matrix using mean and standard deviation
def normalize(X, m, s):
    """Normalize a matrix using mean and standard deviation"""
    return (X - m) / s
