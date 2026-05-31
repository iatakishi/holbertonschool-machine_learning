#!/usr/bin/env python3
"""Module for building a Keras neural network model"""
import tensorflow.keras as K


# Build a neural network model using Keras Sequential API
def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a neural network with L2 regularization and dropout"""
    model = K.Sequential()
    reg = K.regularizers.L2(lambtha)
    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=reg,
                             input_shape=(nx,)))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=reg))
    return model
