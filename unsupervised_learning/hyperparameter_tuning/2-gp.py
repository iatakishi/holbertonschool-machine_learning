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
        sqdist = (X1 - X2.T) ** 2
        K = (self.sigma_f ** 2) * np.exp(-0.5 * sqdist / (self.l ** 2))
        return K

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a GP.

        Args:
            X_s: numpy.ndarray of shape (s, 1) containing all of the points
                 whose mean and standard deviation should be calculated.

        Returns:
            mu, sigma
            - mu is a numpy.ndarray of shape (s,) containing the mean
            - sigma is a numpy.ndarray of shape (s,) containing the variance
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y)
        mu = np.squeeze(mu)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process with a new sample point.

        Args:
            X_new: numpy.ndarray of shape (1,) that represents the new
                   sample point.
            Y_new: numpy.ndarray of shape (1,) that represents the new
                   sample function value.
        """
        # Append the new point to the existing matrices
        # np.vstack cleanly handles reshaping the (1,) input to (1, 1)
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        # Recalculate the covariance kernel matrix with the updated X
        self.K = self.kernel(self.X, self.X)
