#!/usr/bin/env python3
"""
Module to perform pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' for max pooling, 'avg' for average pooling

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions based on stride (no padding in pooling)
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1

    # Initialize the output matrix (channels stay the same, no nc here)
    pooled = np.zeros((m, h_out, w_out, c))

    # Loop through output height and width (only two loops allowed)
    for i in range(h_out):
        for j in range(w_out):
            row_start = i * sh
            col_start = j * sw
            # Extract the current slice across all images, all channels
            image_slice = images[
                :, row_start:row_start + kh,
                col_start:col_start + kw, :
            ]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            else:
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled
