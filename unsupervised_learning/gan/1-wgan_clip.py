#!/usr/bin/env python3
""" WGAN with weight clipping """
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """ WGAN clip class """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """ Init method """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        # Generator wants discriminator to output high values for fake images
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
        # Critic minimizes: mean(fake_preds) - mean(real_preds)
        self.discriminator.loss = lambda real_preds, fake_preds: (
                tf.math.reduce_mean(fake_preds) - tf.math.reduce_mean(real_preds))

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

    def train_step(self, useless_argument):
        """ overloading train_step() """

        # 1. Train the Discriminator (Critic)
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                # get samples
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=False)

                # get predictions
                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)

                # compute the loss discr_loss of the discriminator
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

            # apply gradient descent once to the discriminator
            disc_vars = self.discriminator.trainable_variables
            disc_gradients = disc_tape.gradient(discr_loss, disc_vars)
            self.discriminator.optimizer.apply_gradients(
                zip(disc_gradients, disc_vars))

            # clip the weights of the discriminator between -1 and 1
            for var in disc_vars:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        # 2. Train the Generator
        with tf.GradientTape() as gen_tape:
            # get a fake sample
            fake_samples = self.get_fake_sample(training=True)

            # evaluate fake samples with the discriminator
            fake_preds = self.discriminator(fake_samples, training=False)

            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(fake_preds)

        # apply gradient descent to the generator
        gen_vars = self.generator.trainable_variables
        gen_gradients = gen_tape.gradient(gen_loss, gen_vars)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, gen_vars))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
