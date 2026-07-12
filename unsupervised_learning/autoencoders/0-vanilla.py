#!/usr/bin/env python3
"""
Vanilla Autoencoder module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a vanilla autoencoder model.

    Args:
        input_dims: int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: int containing the dimensions of the latent space

    Returns:
        encoder, decoder, auto
    """
    # --- ENCODER MODEL ---
    encoder_input = keras.Input(shape=(input_dims,))

    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent_output = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent_output, name='encoder')

    # --- DECODER MODEL ---
    decoder_input = keras.Input(shape=(latent_dims,))

    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')

    # --- FULL AUTOENCODER MODEL ---
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output, name='autoencoder')

    # Compile full network
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
