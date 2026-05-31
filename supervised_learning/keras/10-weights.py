#!/usr/bin/env python3
"""Module for saving and loading Keras model weights"""
import tensorflow.keras as K


# Save and load Keras model weights only
def save_weights(network, filename, save_format='keras'):
    """Save a model's weights to a file"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Load weights from a file into a model"""
    network.load_weights(filename)
