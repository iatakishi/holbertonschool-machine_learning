#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer
    of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        activation: activation function applied to the convolution
        padding: 'same' or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Determine padding amounts
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    # Pad the input with zeros on height and width axes
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output (one slot per output channel c_new)
    Z = np.zeros((m, h_new, w_new, c_new))

    # Loop over output height, output width, and each output channel
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                row_start = i * sh
                col_start = j * sw
                # Extract slice across all images and input channels
                a_slice = A_prev_pad[
                    :, row_start:row_start + kh,
                    col_start:col_start + kw, :
                ]
                # Pick the k-th kernel (shape kh, kw, c_prev)
                kernel = W[:, :, :, k]
                # Multiply and sum over height, width, and input channels
                Z[:, i, j, k] = np.sum(
                    a_slice * kernel, axis=(1, 2, 3)
                )

    # Add bias (broadcasts over m, h_new, w_new since b has shape
    # (1,1,1,c_new))
    Z = Z + b

    # Apply the activation function
    A = activation(Z)

    return A
