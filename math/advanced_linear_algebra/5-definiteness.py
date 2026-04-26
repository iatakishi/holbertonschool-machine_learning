#!/usr/bin/env python3
"""Module containing definiteness function"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (matrix.size == 0 or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]):
        return None

    # Compute eigenvalues
    eigs = np.linalg.eigvals(matrix)

    # Definiteness is only defined for matrices with real eigenvalues
    if not np.allclose(eigs.imag, 0, atol=1e-8):
        return None

    eigs = np.real(eigs)
    tol = 1e-8

    # Boolean masks for eigenvalue signs
    pos = eigs > tol
    neg = eigs < -tol
    zero = np.isclose(eigs, 0, atol=tol)

    if np.all(pos):
        return "Positive definite"
    if np.all(neg):
        return "Negative definite"
    if np.all(pos | zero) and np.any(zero):
        return "Positive semi-definite"
    if np.all(neg | zero) and np.any(zero):
        return "Negative semi-definite"
    if np.any(pos) and np.any(neg):
        return "Indefinite"

    return None
