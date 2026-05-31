#!/usr/bin/env python3
"""Module for saving and loading Keras model configuration"""
import tensorflow.keras as K


# Save and load Keras model configuration in JSON format
def save_config(network, filename):
    """Save a model's configuration to a JSON file"""
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """Load a model from a JSON configuration file"""
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
