#!/usr/bin/env python3
"""
Forward propagation for a Deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: A list of RNNCell instances
        of length l used for forward propagation.
                   - l is the number of layers.
        X: The data to be used, given as
        a numpy.ndarray of shape (t, m, i).
           - t is the maximum number of time steps.
           - m is the batch size.
           - i is the dimensionality of the data.
        h_0: The initial hidden state, given as
        a numpy.ndarray of shape (l, m, h).
             - h is the dimensionality of the hidden state.

    Returns:
        H: A numpy.ndarray containing all of the hidden states.
        Y: A numpy.ndarray containing
        all of the outputs from the last layer.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # Initialize the hidden states array
    # H with shape (t + 1, l, m, h)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    Y_list = []

    # Loop over each time step
    for step in range(t):
        # The input for the first layer
        # at the current time step
        x_curr = X[step]

        # Loop over each layer in the deep RNN
        for layer in range(l):
            # Fetch the previous hidden state
            # for the current layer
            h_prev = H[step, layer]

            # Perform forward propagation on the current cell
            h_next, y = rnn_cells[layer].forward(h_prev, x_curr)

            # Store the newly computed hidden state
            H[step + 1, layer] = h_next

            # The hidden state output of the
            # current layer becomes the input
            # for the next layer at the same time step
            x_curr = h_next

        # The output of the entire deep
        # RNN at this time step is the
        # output 'y' from the final layer
        Y_list.append(y)

    # Convert the list of outputs
    # to a numpy array of shape (t, m, o)
    Y = np.array(Y_list)

    return H, Y
