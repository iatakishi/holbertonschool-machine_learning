#!/usr/bin/env python3
"""Module for creating a neural network layer with L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Arguments:
    cost -- tensor containing the cost of the network without L2 regularization
    model -- tensorflow model

    Returns:
    tensor containing the total cost of the network including L2 regularization
    """
    return cost + tf.add_n(model.losses)
