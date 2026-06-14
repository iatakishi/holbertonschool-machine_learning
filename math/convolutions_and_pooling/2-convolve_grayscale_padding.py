#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple of (ph, pw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images with zeros on height and width axes
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Calculate output dimensions based on padded size
    h_out = h + 2 * ph - kh + 1
    w_out = w + 2 * pw - kw + 1

    # Initialize the output matrix with zeros
    convolved = np.zeros((m, h_out, w_out))

    # Loop through output height and width (only two loops allowed)
    for i in range(h_out):
        for j in range(w_out):
            # Extract the current slice across all images at once
            image_slice = padded[:, i:i + kh, j:j + kw]
            # Multiply by kernel and sum over the spatial axes (1 and 2)
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
