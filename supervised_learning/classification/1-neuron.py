#!/usr/bin/env python3
"""Module for Neuron class"""
import numpy as np


# Neuron performing binary classification with private attributes
class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize neuron with nx input features"""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights vector"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A
