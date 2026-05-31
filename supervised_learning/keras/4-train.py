#!/usr/bin/env python3
"""Module for training a Keras model"""
import tensorflow.keras as K


# Train a Keras model using mini-batch gradient descent
def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descent"""
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle)
