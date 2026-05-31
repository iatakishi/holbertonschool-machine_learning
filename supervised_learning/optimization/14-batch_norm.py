#!/usr/bin/env python3
"""Module for creating a batch normalization layer in TensorFlow"""
import tensorflow as tf


# Create a batch normalization layer for a neural network
def create_batch_norm_layer(prev, n, activation):
    """Create a batch normalization layer in TensorFlow"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(n, kernel_initializer=init)(prev)
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    mean, variance = tf.nn.moments(dense, axes=[0])
    Z_norm = tf.nn.batch_normalization(dense, mean, variance,
                                       beta, gamma, 1e-7)
    return activation(Z_norm)
