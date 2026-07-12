#!/usr/bin/env python3
"""
Bayesian Optimization - Optimize
"""
import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor for Bayesian Optimization.

        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
            bounds: a tuple of (min, max) representing the bounds of the space
                    in which to look for the optimal point
            ac_samples: the number of samples that should be analyzed during
                        acquisition
            l: the length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                     black-box function
            xsi: the exploration-exploitation factor for acquisition
            minimize: a bool determining whether optimization should be
                      performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected
        Improvement (EI) acquisition function.

        Returns:
            X_next, EI
            - X_next is a numpy.ndarray of shape (1,) representing the next
              best sample point
            - EI is a numpy.ndarray of shape (ac_samples,) containing the
              expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            imp = mu - Y_opt - self.xsi

        Z = np.zeros(sigma.shape)
        EI = np.zeros(sigma.shape)

        mask = sigma > 0

        Z[mask] = imp[mask] / sigma[mask]

        # Split calculation to bypass W503, W504, and E501 strictness entirely
        term1 = imp[mask] * norm.cdf(Z[mask])
        term2 = sigma[mask] * norm.pdf(Z[mask])
        EI[mask] = term1 + term2

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function by continuously proposing and evaluating
        new points until reaching the maximum iterations or a duplicated point.

        Args:
            iterations: the maximum number of iterations to perform

        Returns:
            X_opt, Y_opt
            - X_opt is a numpy.ndarray of shape (1,) representing the optimal point
            - Y_opt is a numpy.ndarray of shape (1,) representing the optimal
              function value
        """
        for _ in range(iterations):
            # Propose the next best point using the acquisition function
            X_next, _ = self.acquisition()

            # Check if this point has already been sampled to avoid redundancy
            if X_next in self.gp.X:
                break

            # Evaluate the black-box function at the new point
            Y_next = self.f(X_next)

            # Incorporate the new point into the Gaussian Process model
            self.gp.update(X_next, Y_next)

        # Determine the final optimal point based on the objective (min or max)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        # Extract the optimal inputs/outputs (maintaining the (1,) shape)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
