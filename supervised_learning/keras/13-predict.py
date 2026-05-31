#!/usr/bin/env python3
"""Module for making predictions with a Keras model"""
import tensorflow.keras as K


# Make predictions using a Keras neural network model
def predict(network, data, verbose=False):
    """Make a prediction using a neural network"""
    return network.predict(data, verbose=verbose)
