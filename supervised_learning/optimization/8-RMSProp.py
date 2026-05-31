#!/usr/bin/env python3
"""Module for creating a RMSProp optimizer"""
import tensorflow as tf


# Create a RMSProp optimizer using TensorFlow
def create_RMSProp_op(alpha, beta2, epsilon):
    """Create RMSProp optimizer"""
    return tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                       rho=beta2,
                                       epsilon=epsilon)
