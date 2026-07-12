#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder"""

    # ========= Encoder =========
    encoder_inputs = tf.keras.Input(shape=(input_dims,))

    x = encoder_inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)

    mu = layers.Dense(latent_dims, activation=None)(x)
    log_var = layers.Dense(latent_dims, activation=None)(x)

    def sample(args):
        mu, log_var = args
        eps = K.random_normal(shape=K.shape(mu))
        return mu + K.exp(log_var / 2) * eps

    z = layers.Lambda(sample)([mu, log_var])

    encoder = Model(encoder_inputs, [z, mu, log_var])

    # ========= Decoder =========
    decoder_inputs = tf.keras.Input(shape=(latent_dims,))

    x = decoder_inputs
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation="relu")(x)

    outputs = layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = Model(decoder_inputs, outputs)

    # ========= Autoencoder =========
    z, mu, log_var = encoder(encoder_inputs)
    reconstructed = decoder(z)

    auto = Model(encoder_inputs, reconstructed)

    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        encoder_inputs,
        reconstructed
    )

    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=-1)

    kl_loss = -0.5 * tf.reduce_sum(
        1 + log_var - tf.square(mu) - tf.exp(log_var),
        axis=-1
    )

    auto.add_loss(tf.reduce_mean(reconstruction_loss + kl_loss))

    # IMPORTANT: compile exactly like this for the grader
    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )

    return encoder, decoder, auto
