#!/usr/bin/env python3
"""Module for saving and loading Keras models"""
import tensorflow.keras as K


# Save and load entire Keras models
def save_model(network, filename):
    """Save an entire Keras model to a file"""
    network.save(filename)


def load_model(filename):
    """Load an entire Keras model from a file"""
    return K.models.load_model(filename)
