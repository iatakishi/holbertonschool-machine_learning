#!/usr/bin/env python3
"""
Variational Autoencoder module
"""
import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder model."""
    # --- ENCODER ---
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Reparameterization trick"""
        z_m, z_l_v = args
        batch = tf.shape(z_m)[0]
        dim = tf.shape(z_m)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_m + tf.exp(0.5 * z_l_v) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])
    encoder = keras.Model(inputs=encoder_inputs, outputs=[z, z_mean, z_log_var])

    # --- DECODER ---
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    # --- FULL AUTOENCODER ---
    auto_inputs = keras.Input(shape=(input_dims,))
    # Destructure the encoder outputs to only pass 'z' to the decoder
    encoded_z, _, _ = encoder(auto_inputs)
    auto_outputs = decoder(encoded_z)
    auto = keras.Model(inputs=auto_inputs, outputs=auto_outputs)

    # --- CUSTOM LOSS ---
    def vae_loss(y_true, y_pred):
        recon_loss = keras.losses.binary_crossentropy(y_true, y_pred) * input_dims
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var),
            axis=-1
        )
        return recon_loss + kl_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
