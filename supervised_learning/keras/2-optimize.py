#!/usr/bin/env python3
"""Module for setting up Adam optimization for a Keras model"""
import tensorflow.keras as K


# Set up Adam optimization for a Keras model
def optimize_model(network, alpha, beta1, beta2):
    """Compile the model with Adam optimizer
    and categorical cross-entropy loss"""
    network.compile(
        optimizer=K.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1,
                                    beta_2=beta2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
