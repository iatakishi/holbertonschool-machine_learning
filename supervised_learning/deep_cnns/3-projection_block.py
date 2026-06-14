#!/usr/bin/env python3
"""
Module containing the projection_block function for a ResNet architecture.
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning
    for Image Recognition (2015).

    Args:
        A_prev: tensor output from the previous layer
        filters: tuple or list containing F11, F3, F12 respectively:
            F11: number of filters in the first 1x1 convolution
            F3: number of filters in the 3x3 convolution
            F12: number of filters in the second 1x1 convolution as well
                 as the 1x1 convolution in the shortcut connection
        s: stride of the first convolution in both the main path and
           the shortcut connection

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters

    # He normal initializer with fixed seed 0
    initializer = K.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='normal', seed=0
    )

    # ==========================================
    #               MAIN PATH
    # ==========================================

    # --- FIRST COMPONENT (1x1 Bottleneck with stride s) ---
    x = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)

    # --- SECOND COMPONENT (3x3 Convolution) ---
    x = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)

    # --- THIRD COMPONENT (1x1 Bottleneck expansion) ---
    x = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(x)
    x = K.layers.BatchNormalization(axis=-1)(x)

    # ==========================================
    #            SHORTCUT PATH
    # ==========================================
    # We apply a 1x1 conv to A_prev to project its dimensions to match 'x'
    shortcut = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=-1)(shortcut)

    # ==========================================
    #         MERGE SHORTCUT & MAIN PATH
    # ==========================================
    x = K.layers.Add()([x, shortcut])

    # Final ReLU Activation
    x = K.layers.Activation('relu')(x)

    return x
