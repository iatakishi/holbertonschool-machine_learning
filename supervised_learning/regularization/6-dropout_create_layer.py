#!/usr/bin/env python3
"""
Creates a layer of a neural network using dropout.
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout:

    * prev: A tensor containing the output of the previous layer
    * n: The number of nodes the new layer should contain
    * activation: The activation function for the new layer
    * keep_prob: The probability that a node will be kept
    * training: A boolean indicating whether the model is in training mode

    Returns: The output of the new layer
    """
    # Standard initialization for Holberton ML tasks
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    # Create the dense layer
    dense_layer = tf.keras.layers.Dense(units=n,
                                        activation=activation,
                                        kernel_initializer=initializer)

    # Generate the output from the dense layer
    layer_output = dense_layer(prev)

    # Create the Dropout layer. Note: Keras Dropout expects 'rate' (fraction to drop),
    # so we must convert keep_prob to rate by subtracting it from 1.0.
    dropout_layer = tf.keras.layers.Dropout(rate=1.0 - keep_prob)

    # Apply dropout to the output, explicitly passing the training flag
    return dropout_layer(layer_output, training=training)
