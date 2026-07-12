#!/usr/bin/env python3
"""
Convolutional Autoencoder module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder model.

    Args:
        input_dims: tuple of integers containing the dimensions of the
                    model input
        filters: list containing the number of filters for each convolutional
                 layer in the encoder, respectively
        latent_dims: tuple of integers containing the dimensions of the
                     latent space representation

    Returns:
        encoder, decoder, auto
    """
    # --- ENCODER MODEL ---
    encoder_input = keras.Input(shape=input_dims)

    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(
            f, (3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoder = keras.Model(encoder_input, x, name='encoder')

    # --- DECODER MODEL ---
    decoder_input = keras.Input(shape=latent_dims)

    x = decoder_input
    dec_filters = list(reversed(filters))

    # Process all layers except the very last one
    for i in range(len(dec_filters)):
        if i == len(dec_filters) - 1:
            # The second to last convolution layer uses valid padding
            x = keras.layers.Conv2D(
                dec_filters[i], (3, 3),
                padding='valid',
                activation='relu'
            )(x)
        else:
            # Standard decoder layers use same padding
            x = keras.layers.Conv2D(
                dec_filters[i], (3, 3),
                padding='same',
                activation='relu'
            )(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    # The last convolution layer uses sigmoid activation and no upsampling
    decoder_output = keras.layers.Conv2D(
        input_dims[-1], (3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, decoder_output, name='decoder')

    # --- FULL AUTOENCODER MODEL ---
    auto_output = decoder(encoder(encoder_input))
    auto = keras.Model(encoder_input, auto_output, name='autoencoder')

    # Compile full network
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
