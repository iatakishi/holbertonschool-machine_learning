#!/usr/bin/env python3
"""Module for calculating L2 regularization cost with Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost without L2 regularization
        model: Keras model with layers that include L2 regularization

    Returns:
        tensor containing the total cost for each layer accounting for L2
    """
    l2_costs = [
        cost + layer.losses
        for layer in model.layers
        if layer.losses
    ]
    return tf.cast(l2_costs, dtype=tf.float32)
