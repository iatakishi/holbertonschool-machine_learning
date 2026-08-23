#!/usr/bin/env python3
""" WGAN with gradient penalty and weight replacement """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """ WGAN GP class """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """ Init method """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda fake_preds: -tf.math.reduce_mean(
            fake_preds)

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda real_preds, fake_preds: (
                tf.math.reduce_mean(fake_preds)
                - tf.math.reduce_mean(real_preds))

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """ generator of fake samples """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """ generator of real samples """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """ generator of interpolating samples """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """ computing the gradient penalty """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """ overloading train_step() """

        # 1. Train Discriminator
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                # get samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=False)
                interp_samples = self.get_interpolated_sample(
                    real_samples, fake_samples)

                # get predictions
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)

                # compute discriminator loss
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

                # compute the gradient penalty gp
                gp = self.gradient_penalty(interp_samples)

                # compute the combined discriminator loss
                new_discr_loss = discr_loss + self.lambda_gp * gp

            # apply gradient descent
            disc_vars = self.discriminator.trainable_variables
            disc_gradients = disc_tape.gradient(new_discr_loss, disc_vars)
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradients, disc_vars))

        # 2. Train Generator
        with tf.GradientTape() as gen_tape:
            # get fake sample
            fake_samples = self.get_fake_sample(training=True)

            # evaluate with discriminator
            fake_preds = self.discriminator(fake_samples, training=False)

            # compute generator loss
            gen_loss = self.generator.loss(fake_preds)

        # apply gradient descent
        gen_vars = self.generator.trainable_variables
        gen_gradients = gen_tape.gradient(gen_loss, gen_vars)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, gen_vars))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """ Replaces the weights of the generator and discriminator """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
