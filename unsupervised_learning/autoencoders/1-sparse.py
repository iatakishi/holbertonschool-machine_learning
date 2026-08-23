#!/usr/bin/env python3
"""
Sparse Autoencoder module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder model with L1 regularization on the
    encoded output.

    Args:
        input_dims: int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively
        latent_dims: int containing the dimensions of the latent space
        lambtha: regularization parameter used for L1 regularization on
                 the encoded output

    Returns:
        encoder, decoder, auto
    """
    # --- ENCODER MODEL ---
    encoder_input = keras.Input(shape=(input_dims,))

    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Apply L1 activity regularization to penalize non-zero activations
    latent_output = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
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
