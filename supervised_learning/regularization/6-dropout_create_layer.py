#!/usr/bin/env python3
"""Module for creating a layer with dropout in TensorFlow."""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Arguments:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function for the new layer
    keep_prob -- probability that a node will be kept
    training -- boolean indicating
    if the model is in training mode

    Returns:
    the output of the new layer
    """
    dense = tf.keras.layers.Dense(
        units=n,
        activation=activation
    )
    dropout = tf.keras.layers.Dropout(
        rate=1 - keep_prob
    )
    return dropout(dense(prev), training=training)
