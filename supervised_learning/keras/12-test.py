#!/usr/bin/env python3
"""Module for testing a Keras model"""
import tensorflow.keras as K


# Test a Keras neural network model
def test_model(network, data, labels, verbose=True):
    """Test a neural network and return loss and accuracy"""
    return network.evaluate(data, labels, verbose=verbose)
