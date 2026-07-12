#!/usr/bin/env python3
"""
Variational Autoencoder module
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder model.

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

    # VAE Bottleneck projections without activation (None)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization Trick to allow backpropagation
    def sampling(args):
        """Samples from the latent normal distribution using mu and log_var"""
        mu, log_var = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(0.5 * log_var) * epsilon

    # Lambda layer transforms projections into the latent representation tensor
    z = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([z_mean, z_log_var])

    encoder = keras.Model(
        encoder_input,
        [z, z_mean, z_log_var],
        name='encoder'
    )

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

    # --- FULL VARIATIONAL AUTOENCODER MODEL ---
    auto_output = decoder(encoder(encoder_input)[0])
    auto = keras.Model(encoder_input, auto_output, name='autoencoder')

    # --- CUSTOM VAE LOSS FUNCTION ---
    def vae_loss(y_true, y_pred):
        """Calculates total loss = reconstruction loss + KL divergence"""
        # Reconstruction Loss (scaled up by input dimensions)
        recon_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        recon_loss *= input_dims

        # KL Divergence Loss regularizes latent distribution to standard normal
        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var - keras.backend.square(z_mean) -
            keras.backend.exp(z_log_var),
            axis=-1
        )
        return keras.backend.mean(recon_loss + kl_loss)

    # Compile network using custom VAE loss
    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
