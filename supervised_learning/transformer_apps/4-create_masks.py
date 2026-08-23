#!/usr/bin/env python3
"""
Module to create masks for training and validation in a Transformer.
"""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in).
        target: tf.Tensor of shape (batch_size, seq_len_out).

    Returns:
        encoder_mask: tf.Tensor padding mask of shape
                      (batch_size, 1, 1, seq_len_in).
        combined_mask: tf.Tensor of shape
                       (batch_size, 1, seq_len_out, seq_len_out).
        decoder_mask: tf.Tensor padding mask of shape
                      (batch_size, 1, 1, seq_len_in).
    """
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask (same as encoder mask)
    decoder_mask = encoder_mask

    # Look-ahead mask (masks future tokens)
    seq_len = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len)), -1, 0
    )

    # Target padding mask
    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]

    # Combined mask (maximum of look-ahead and target padding)
    combined_mask = tf.maximum(target_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
