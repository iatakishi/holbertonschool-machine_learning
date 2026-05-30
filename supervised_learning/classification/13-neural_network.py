#!/usr/bin/env python3
"""Module for NeuralNetwork class"""
import numpy as np


# Neural network with one hidden layer and gradient descent
class NeuralNetwork:
    """Defines a neural network with one
    hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Initialize neural network
        with nx inputs and nodes in hidden layer"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for hidden layer weights"""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer bias"""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output"""
        return self.__A1

    @property
    def W2(self):
        """Getter for output neuron weights"""
        return self.__W2

    @property
    def b2(self):
        """Getter for output neuron bias"""
        return self.__b2

    @property
    def A2(self):
        """Getter for output neuron activated output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculate forward propagation through both layers"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate logistic regression cost"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions and cost"""
        _, A2 = self.forward_prop(X)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
