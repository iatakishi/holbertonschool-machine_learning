#!/usr/bin/env python3
"""
Module to perform back propagation over a convolutional layer.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer
    of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new), gradients
            with respect to the unactivated output of this layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        padding: 'same' or 'valid'
        stride: tuple of (sh, sw)

    Returns:
        dA_prev, dW, db
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Determine padding amounts (same logic as conv_forward)
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Pad A_prev the same way as in conv_forward
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Initialize gradients with zeros, same shapes as inputs
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    # db: sum dZ over m, h_new, w_new -> shape (1,1,1,c_new)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Loop over output height, output width, and each output channel
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                row_start = i * sh
                col_start = j * sw

                # Slice of the padded input that produced Z[:, i, j, k]
                a_slice = A_prev_pad[
                    :, row_start:row_start + kh,
                    col_start:col_start + kw, :
                ]

                # dZ for this output position/channel, shape (m,)
                # reshape to (m,1,1,1) for broadcasting
                dz = dZ[:, i, j, k].reshape(m, 1, 1, 1)

                # Accumulate gradient w.r.t. the input slice:
                # each input pixel contributed via W[:,:,:,k],
                # scaled by dz for each example
                dA_prev_pad[
                    :, row_start:row_start + kh,
                    col_start:col_start + kw, :
                ] += W[:, :, :, k] * dz

                # Accumulate gradient w.r.t. the kernel:
                # sum over all m examples of (a_slice * dz)
                dW[:, :, :, k] += np.sum(a_slice * dz, axis=0)

    # Remove padding from dA_prev_pad to get dA_prev
    if padding == 'same':
        dA_prev = dA_prev_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
