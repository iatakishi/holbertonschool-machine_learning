#!/usr/bin/env python3
"""
Module that contains the RNNCell class
"""
import numpy as np


class RNNCell:
    """
    Represents a single cell of a simple Recurrent Neural Network
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        # Wh is for the concatenated hidden state and input data
        # Since we concatenate h_prev (shape m, h) and x_t (shape m, i) along axis 1
        # the concatenated shape is (m, h + i). Thus, Wh needs shape (h + i, h)
        self.Wh = np.random.randn(h + i, h)

        # Wy is for the output y (computed from the hidden state)
        self.Wy = np.random.randn(h, o)

        # Biases initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
        h_prev is a numpy.ndarray of shape (m, h) containing the previous hidden state
        x_t is a numpy.ndarray of shape (m, i) containing the data input for the cell
            m is the batch size for the data

        Returns: h_next, y
        h_next is the next hidden state
        y is the output of the cell
        """
        # Concatenate previous hidden state and current input
        # Shape: (m, h + i)
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Compute the next hidden state using tanh activation
        # Shape: (m, h)
        h_next = np.tanh(np.matmul(h_x, self.Wh) + self.bh)

        # Compute the output using softmax activation
        # Shape: (m, o)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
