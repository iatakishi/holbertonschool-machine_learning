#!/usr/bin/env python3
"""Module for training a Keras model with learning rate decay"""
import tensorflow.keras as K


# Train a Keras model with optional learning rate decay
def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Train a model with early stopping and learning rate decay"""
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))
    if learning_rate_decay and validation_data:
        def schedule(epoch):
            """Inverse time decay learning rate schedule"""
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(K.callbacks.LearningRateScheduler(schedule,
                                                           verbose=1))
    return network.fit(data, labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
