#!/usr/bin/env python3
"""
Module that performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Parameters:
    rnn_cell is an instance of RNNCell
    that will be used for the forward prop
    X is the data to be used, given
    as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state,
    given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state

    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    # We can infer the output dimensionality
    # (o) from the bias of the cell
    o = rnn_cell.by.shape[1]

    # Initialize the H and Y arrays with zeros
    # H has shape (t + 1, m, h) to include
    # the initial hidden state h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set the initial hidden state
    H[0] = h_0

    # Loop through each time step
    for step in range(t):
        # Extract the input for the current time step
        x_t = X[step]

        # Perform forward propagation for this step
        h_next, y_t = rnn_cell.forward(H[step], x_t)

        # Store the updated hidden state and the output
        H[step + 1] = h_next
        Y[step] = y_t

    return H, Y
