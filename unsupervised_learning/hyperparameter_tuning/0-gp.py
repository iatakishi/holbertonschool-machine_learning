#!/usr/bin/env python3
"""
Initialize a noiseless 1D Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor for the Gaussian Process.

        Args:
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function.
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init.
            l: the length parameter for the kernel.
            sigma_f: the standard deviation given to the output of the
                     black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF).

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            The covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        # Since this is a 1D Gaussian Process, we can compute the squared
        # Euclidean distance simply by broadcasting the difference
        sqdist = (X1 - X2.T) ** 2

        # Calculate the RBF kernel matrix
        K = (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))

        return K
