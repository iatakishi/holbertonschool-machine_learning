#!/usr/bin/env python3
""" Convolutional Generator and Discriminator for GANs """
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional generator and discriminator
    Returns: gen (keras.Model), discr (keras.Model)
    """

    def get_generator():
        """ Builds the generator model """
        inputs = keras.Input(shape=(16,))

        # Dense + Reshape
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Block 1
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 2
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(16, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 3
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(1, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('tanh')(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        """ Builds the discriminator model """
        inputs = keras.Input(shape=(16, 16, 1))

        # Block 1
        x = keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 2
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 3
        x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Block 4
        x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation('tanh')(x)

        # Flatten + Dense
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return get_generator(), get_discriminator()
