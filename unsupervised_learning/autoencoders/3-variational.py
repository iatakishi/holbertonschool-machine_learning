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
    encoder = keras.Model(inputs=encoder_input, outputs=latent_output)

    # --- DECODER MODEL ---
    decoder_input = keras.Input(shape=(latent_dims,))

    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_input, outputs=decoder_output)

    # --- FULL AUTOENCODER MODEL ---
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(inputs=encoder_input, outputs=auto_output)

    # Compile network explicitly using the string alias
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
