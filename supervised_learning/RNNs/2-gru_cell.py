#!/usr/bin/env python3
"""
Module that contains the GRUCell class
"""
import numpy as np


class GRUCell:
    """
    Represents a Gated Recurrent Unit (GRU).
    """

    def __init__(self, i, h, o):
        """
        Class constructor for the GRU cell.

        Args:
            i (int): Dimensionality of the data
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Initialize weights with random
        # normal distribution
        # Weights are concatenated
        # for matrix multiplication on the right: (h + i, h)
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)

        # Output weights
        self.Wy = np.random.randn(h, o)

        # Initialize biases as zeros
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Array of shape
            (m, h) containing the previous hidden state
            x_t (numpy.ndarray): Array of shape
            (m, i) containing the data input for the cell
                where m is the batch size for the data

        Returns:
            h_next, y
                - h_next is the next hidden state
                - y is the output of the cell
        """
        # Concatenate the previous hidden state and
        # input data for matrix multiplication
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Update Gate (z)
        z_linear = np.matmul(h_x, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z_linear))  # Sigmoid activation

        # Reset Gate (r)
        r_linear = np.matmul(h_x, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r_linear))  # Sigmoid activation

        # Intermediate Hidden State (h_tilde)
        # Apply reset gate to previous hidden state,
        # then concatenate with input
        r_h_prev = r * h_prev
        r_h_x = np.concatenate((r_h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(r_h_x, self.Wh) + self.bh)

        # Next Hidden State (h_next)
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output (y) with Softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        # Shift values for numerical stability in Softmax
        e_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = e_y / np.sum(e_y, axis=1, keepdims=True)

        return h_next, y
