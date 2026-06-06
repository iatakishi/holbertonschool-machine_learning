#!/usr/bin/env python3
"""Module for creating a neural network layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    Args:
        prev: tensor containing output of previous layer
        n: number of nodes in the new layer
        activation: activation function to apply to the layer
        lambtha: L2 regularization parameter

    Returns:
        The output tensor of the new layer
    """
    # Glorot uniform initializer with seed=0 for reproducibility
    initializer = tf.keras.initializers.glorot_uniform(seed=0)

    # L2 regularizer with given lambda
    regularizer = tf.keras.regularizers.l2(lambtha)

    # Create the Dense layer with L2 regularization on the kernel
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    # Apply the layer to the previous layer's output
    return layer(prev)
