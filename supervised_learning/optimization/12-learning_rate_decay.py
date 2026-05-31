#!/usr/bin/env python3
"""Module for learning rate decay using TensorFlow"""
import tensorflow as tf


# Create learning rate decay operation using TensorFlow
def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a learning rate decay operation using inverse time decay"""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
