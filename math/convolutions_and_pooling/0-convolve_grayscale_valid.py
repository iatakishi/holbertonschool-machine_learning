#!/usr/bin/env python3
"""
Module to perform a valid convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions for 'valid' convolution
    h_out = h - kh + 1
    w_out = w - kw + 1

    # Initialize the output matrix with zeros
    convolved = np.zeros((m, h_out, w_out))

    # Loop through the output height and width (only two loops allowed)
    for i in range(h_out):
        for j in range(w_out):
            # Extract the current slice across all images at once
            image_slice = images[:, i:i + kh, j:j + kw]
            # Multiply by kernel and sum over the spatial axes (1 and 2)
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
