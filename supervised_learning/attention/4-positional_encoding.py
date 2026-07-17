#!/usr/bin/env python3
"""Calculates the positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    Args:
        max_seq_len: integer representing the maximum sequence length
        dm: the model depth

    Returns:
        a numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))

    positions = np.arange(max_seq_len)[:, np.newaxis]
    dims = np.arange(dm)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(dm))
    angles = positions * angle_rates

    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE
