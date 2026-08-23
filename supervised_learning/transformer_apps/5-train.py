#!/usr/bin/env python3
"""Training script for the Transformer model."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the schedule."""
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Calculate learning rate."""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Calculate sparse categorical crossentropy ignoring padding."""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Calculate accuracy ignoring padding."""
    accuracies = tf.equal(
        real, tf.cast(tf.argmax(pred, axis=2, output_type=tf.int32), tf.int32))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation.

    Args:
        N: Number of blocks in encoder and decoder.
        dm: Dimensionality of the model.
        h: Number of heads.
        hidden: Number of hidden units in FFN.
        max_len: Maximum number of tokens per sequence.
        batch_size: Batch size for training.
        epochs: Number of epochs to train.

    Returns:
        The trained Transformer model.
    """
    dataset = Dataset(batch_size, max_len)
    transformer = Transformer(N, dm, h, hidden, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inputs, target)) in enumerate(dataset.data_train):
            tar_inp = target[:, :-1]
            tar_real = target[:, 1:]

            enc_mask, combined_mask, dec_mask = create_masks(
                inputs, tar_inp)

            with tf.GradientTape() as tape:
                predictions, _ = transformer(
                    inputs, tar_inp, True, enc_mask, combined_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(
                loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(tar_real, predictions)

            if batch % 50 == 0:
                print("Epoch {}, Batch {}: Loss {}, Accuracy {}".format(
                    epoch + 1, batch,
                    train_loss.result(),
                    train_accuracy.result()))

        print("Epoch {}: Loss {}, Accuracy {}".format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result()))

    return transformer
