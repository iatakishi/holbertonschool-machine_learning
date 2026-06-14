#!/usr/bin/env python3
"""
Module to perform a strided convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding
    and stride.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: 'same', 'valid', or a tuple of (ph, pw)
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding amounts based on the padding argument
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images with zeros on height and width axes
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Calculate output dimensions based on padded size and stride
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1

    # Initialize the output matrix with zeros
    convolved = np.zeros((m, h_out, w_out))

    # Loop through output height and width (only two loops allowed)
    for i in range(h_out):
        for j in range(w_out):
            # Map output position to padded image position using stride
            row_start = i * sh
            col_start = j * sw
            # Extract the current slice across all images at once
            image_slice = padded[
                :, row_start:row_start + kh, col_start:col_start + kw
            ]
            # Multiply by kernel and sum over the spatial axes (1 and 2)
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
