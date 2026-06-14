#!/usr/bin/env python3
"""
Module containing a function to randomly change image brightness.
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor c
        ontaining the image to change.
        max_delta: The maximum amount
        the image should be brightened
                   or darkened.

    Returns:
        The brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(image, max_delta)
