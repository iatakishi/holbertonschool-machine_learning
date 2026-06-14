#!/usr/bin/env python3
"""
Module to perform a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding needed on each side so output size == input size
    ph = kh // 2
    pw = kw // 2

    # Pad images with zeros on height and width axes
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output has same dimensions as original images
    convolved = np.zeros((m, h, w))

    # Loop through output height and width (only two loops allowed)
    for i in range(h):
        for j in range(w):
            # Extract the current slice across all images at once
            image_slice = padded[:, i:i + kh, j:j + kw]
            # Multiply by kernel and sum over the spatial axes (1 and 2)
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
