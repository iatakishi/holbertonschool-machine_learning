#!/usr/bin/env python3
"""
Module to perform forward propagation over a pooling layer.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer
    of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions (no padding in pooling)
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    # Initialize output (channels stay the same as input, c_prev)
    A = np.zeros((m, h_new, w_new, c_prev))

    # Loop through output height and width (only two loops allowed)
    for i in range(h_new):
        for j in range(w_new):
            row_start = i * sh
            col_start = j * sw
            # Extract the current slice across all images, all channels
            a_slice = A_prev[
                :, row_start:row_start + kh,
                col_start:col_start + kw, :
            ]
            if mode == 'max':
                A[:, i, j, :] = np.max(a_slice, axis=(1, 2))
            else:
                A[:, i, j, :] = np.mean(a_slice, axis=(1, 2))

    return A
