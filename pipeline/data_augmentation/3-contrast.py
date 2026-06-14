#!/usr/bin/env python3
"""
Module containing a function to randomly adjust image contrast.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor containing the image to adjust.
        lower: A float representing
        the lower bound of the contrast factor.
        upper: A float representing
        the upper bound of the contrast factor.

    Returns:
        The contrast-adjusted image tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
