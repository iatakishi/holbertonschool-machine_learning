#!/usr/bin/env python3
"""Module for building a Keras neural network model using Input"""
import tensorflow.keras as K


# Build a neural network model using Keras functional API
def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with L2 regularization and dropout"""
    reg = K.regularizers.L2(lambtha)
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(layers[0], activation=activations[0], kernel_regularizer=reg)(inputs)
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=reg)(x)
    return K.Model(inputs=inputs, outputs=x)
