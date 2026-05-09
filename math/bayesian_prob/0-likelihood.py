#!/usr/bin/env python3
""" likelihood """
import numpy as np


def likelihood(x, n, P):
    """ likelihood """
    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if x < 0 or not isinstance(x, int):
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P <0):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(n):
        f = 1
        for i in range(1, n + 1):
            f *= i
        return f
    combination = factorial(n) / (factorial(x) * factorial(n - x))
    return combination * (P ** x) * ((1 - P) ** (n - x))
