#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def autoencoder(input_dims, hidden_layers, latent_dims):
    # --- Encoder ---
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    # Hidden layers for encoder
    for nodes in hidden_layers:
        x = layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance layers (activation=None)
    z_mean = layers.Dense(latent_dims, activation=None, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dims, activation=None, name='z_log_var')(x)

    # Reparameterization trick (Sampling)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Encoder model outputs: latent representation, mean, and log variance
    encoder = keras.Model(inputs=encoder_inputs, outputs=[z, z_mean, z_log_var], name='encoder')

    # --- Decoder ---
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Hidden layers for decoder (reversed)
    for nodes in reversed(hidden_layers):
        x = layers.Dense(nodes, activation='relu')(x)

    # Output layer for decoder (activation='sigmoid')
    decoder_outputs = layers.Dense(input_dims, activation='sigmoid')(x)

    # Decoder model definition
    decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name='decoder')

    # --- Full Autoencoder ---
    # Connect encoder and decoder (we only pass the sampled 'z' to the decoder)
    auto_outputs = decoder(encoder(encoder_inputs)[0])

    auto = keras.Model(inputs=encoder_inputs, outputs=auto_outputs, name='autoencoder')

    # Compile using exact string identifiers
    # Use lowercase 'adam' and 'binary_crossentropy' as strings
    auto.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
