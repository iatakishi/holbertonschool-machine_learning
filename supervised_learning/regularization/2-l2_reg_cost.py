#!/usr/bin/env python3
"""Module for calculating L2 regularization cost with Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Adds L2 regularization losses to the base cost."""
    return cost + tf.reduce_sum(model.losses)
