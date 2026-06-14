#!/usr/bin/env python3
"""
Module containing a function to flip an image horizontally.
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: A 3D tf.Tensor containing the image to flip.

    Returns:
        The horizontally flipped image tensor.
    """
    return tf.image.flip_left_right(image)
