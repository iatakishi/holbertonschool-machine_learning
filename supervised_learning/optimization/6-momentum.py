#!/usr/bin/env python3
"""Module for creating a momentum optimizer"""
import tensorflow as tf


# Create a momentum optimizer using TensorFlow
def create_momentum_op(alpha, beta1):
    """Create gradient descent with momentum optimizer"""
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
