#!/usr/bin/env python3
"""Module for creating an Adam optimizer"""
import tensorflow as tf


# Create an Adam optimizer using TensorFlow
def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Create Adam optimizer"""
    return tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2,
                                    epsilon=epsilon)
