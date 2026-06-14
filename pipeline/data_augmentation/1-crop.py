#!/usr/bin/env python3
"""
Module containing a function to randomly crop an image.
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop.
        size: A tuple containing the size of the crop (height, width, channels).

    Returns:
        The randomly cropped image tensor.
    """
    return tf.image.random_crop(image, size)
