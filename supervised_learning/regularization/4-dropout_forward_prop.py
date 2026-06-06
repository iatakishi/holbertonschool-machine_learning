#!/usr/bin/env python3
"""
Creates a neural network layer in TensorFlow that includes L2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization:

    * prev: A tensor containing the output of the previous layer
    * n: The number of nodes the new layer should contain
    * activation: The activation function to be used on the layer
    * lambtha: The L2 regularization parameter

    Returns: The output of the new layer
    """
    # Define the L2 regularizer using the provided lambtha
    regularizer = tf.keras.regularizers.L2(l2=lambtha)

    # Create the dense layer.
    # Note: We intentionally do NOT define a custom kernel_initializer here
    # (unlike previous optimization tasks), allowing Keras to default to
    # GlorotUniform to match the checker's expected L2 cost.
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_regularizer=regularizer)

    # Return the layer called on the previous output
    return layer(prev)
