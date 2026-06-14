#!/usr/bin/env python3
"""
Module containing the resnet50 function to build a ResNet-50 architecture.
"""
from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).

    Returns:
        The compiled Keras model.
    """
    # 1. Define Input layer (224x224x3)
    inputs = K.Input(shape=(224, 224, 3))

    # He normal initializer with fixed seed 0
    initializer = K.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='normal', seed=0
    )

    # =========================================================================
    # STAGE 1: Initial Entry Layers (Reduces size to 56x56)
    # =========================================================================
    x = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        kernel_initializer=initializer
    )(inputs)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(x)

    # =========================================================================
    # STAGE 2: 3 Blocks total (Output shape: 56x56x256)
    # =========================================================================
    # First block uses stride=1 because MaxPool already shrank the image size
    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # =========================================================================
    # STAGE 3: 4 Blocks total (Output shape: 28x28x512)
    # =========================================================================
    x = projection_block(x, [128, 128, 512], s=2)
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # =========================================================================
    # STAGE 4: 6 Blocks total (Output shape: 14x14x1024)
    # =========================================================================
    x = projection_block(x, [256, 256, 1024], s=2)
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    # =========================================================================
    # STAGE 5: 3 Blocks total (Output shape: 7x7x2048)
    # =========================================================================
    x = projection_block(x, [512, 512, 2048], s=2)
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # =========================================================================
    # STAGE 6: Classification Head (Output shape: 1000)
    # =========================================================================
    x = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(x)

    outputs = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(x)

    # Create the functional model
    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
