#!/usr/bin/env python3
"""
Module to perform a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: 'same', 'valid', or a tuple of (ph, pw)
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Determine padding amounts based on the padding argument
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images with zeros on height and width axes (not channels)
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions based on padded size and stride
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    # Initialize the output matrix (one extra dim for number of kernels)
    convolved = np.zeros((m, h_out, w_out, nc))

    # Loop over output height, output width, and each kernel
    for i in range(h_out):
        for j in range(w_out):
            for k in range(nc):
                row_start = i * sh
                col_start = j * sw
                # Extract the current slice across all images, channels
                image_slice = padded[
                    :, row_start:row_start + kh,
                    col_start:col_start + kw, :
                ]
                # Pick the k-th kernel
                current_kernel = kernels[:, :, :, k]
                # Multiply and sum over height, width, and channels
                convolved[:, i, j, k] = np.sum(
                    image_slice * current_kernel, axis=(1, 2, 3)
                )

    return convolved
