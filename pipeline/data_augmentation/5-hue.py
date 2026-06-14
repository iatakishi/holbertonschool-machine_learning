#!/usr/bin/env python3
"""
Module containing a function to change the hue of an image.
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: A 3D tf.Tensor containing
        the image to change.
        delta: A float representing
        the amount the hue should change.

    Returns:
        The hue-adjusted image tensor.
    """
    return tf.image.adjust_hue(image, delta)
