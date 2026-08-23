#!/usr/bin/env python3
"""
Module that contains the LSTMCell class
"""

import numpy as np


class LSTMCell:
    """
    Represents an LSTM (Long Short-Term Memory) unit.
    """

    def __init__(self, i, h, o):
        """
        Constructor for the LSTMCell class.

        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        """
        # The concatenated matrix will have shape (m, h + i)
        # Weights are used on the right side for matrix multiplication
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized to zeros
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev is a numpy.ndarray of shape
            (m, h) containing the
                   previous hidden state
            c_prev is a numpy.ndarray of shape
            (m, h) containing the
                   previous cell state
            x_t is a numpy.ndarray of shape
            (m, i) containing the data
                input for the cell
                - m is the batch size for the data

        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        # Concatenate previous hidden state and input data
        # Shape becomes (m, h + i)
        hx = np.concatenate((h_prev, x_t), axis=1)

        # Forget Gate
        f_z = np.matmul(hx, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f_z))

        # Update (Input) Gate
        u_z = np.matmul(hx, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u_z))

        # Intermediate Cell State
        c_z = np.matmul(hx, self.Wc) + self.bc
        c_tilde = np.tanh(c_z)

        # Next Cell State
        c_next = f * c_prev + u * c_tilde

        # Output Gate
        o_z = np.matmul(hx, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o_z))

        # Next Hidden State
        h_next = o * np.tanh(c_next)

        # Output (y) calculation with Softmax activation
        y_z = np.matmul(h_next, self.Wy) + self.by

        # Softmax function
        # (shifting values for numerical stability)
        y_exp = np.exp(y_z - np.max(y_z, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, c_next, y
