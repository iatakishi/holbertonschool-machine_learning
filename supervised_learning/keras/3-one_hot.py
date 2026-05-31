#!/usr/bin/env python3
"""Module for one-hot encoding labels"""
import tensorflow.keras as K


# Convert label vector to one-hot matrix
def one_hot(labels, classes=None):
    """Convert a label vector into a one-hot matrix"""
    return K.utils.to_categorical(labels, num_classes=classes)
