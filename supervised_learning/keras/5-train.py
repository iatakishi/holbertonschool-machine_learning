#!/usr/bin/env python3
"""Module for training a Keras model with validation"""
import tensorflow.keras as K


# Train a Keras model with optional validation data
def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Train a model using mini-batch
    gradient descent with optional validation"""
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data)
