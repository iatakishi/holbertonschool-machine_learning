#!/usr/bin/env python3
"""
Contains the lenet5 function that builds a modified LeNet-5 using Keras.
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras.

    Parameters:
    - X: K.Input of shape (m, 28, 28, 1) containing the input images

    Returns:
    - A K.Model compiled to use Adam optimization and accuracy metrics
    """
    # Initialize weights using he_normal with a fixed seed of 0
    init = K.initializers.HeNormal(seed=0)

    # Layer 1: Conv 5x5, 6 kernels, same padding, relu activation
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Layer 2: Max pooling 2x2, stride 2x2
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Layer 3: Conv 5x5, 16 kernels, valid padding, relu activation
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # Layer 4: Max pooling 2x2, stride 2x2
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten layer to pass data from tensor to fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # Layer 5: Fully connected layer with 120 nodes, relu activation
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # Layer 6: Fully connected layer with 84 nodes, relu activation
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(fc2)

    # Construct the functional Keras model
    model = K.Model(inputs=X, outputs=output)

    # Compile using Adam optimizer and crossentropy loss
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
