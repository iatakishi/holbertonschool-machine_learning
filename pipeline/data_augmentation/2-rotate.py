#!/usr/bin/env python3
"""
Module containing a function to rotate an image.
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    Args:
        image: A 3D tf.Tensor containing the image to rotate.

    Returns:
        The rotated image tensor.
    """
    return tf.image.rot90(image, k=1)
