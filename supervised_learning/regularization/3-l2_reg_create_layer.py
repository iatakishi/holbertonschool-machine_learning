#!/usr/bin/env python3
"""Module for creating a neural network layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in tensorflow that includes L2 regularization.

    Arguments:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function that should be used on the layer
    lambtha -- L2 regularization parameter

    Returns:
    the output of the new layer
    """
    return tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )(prev)
