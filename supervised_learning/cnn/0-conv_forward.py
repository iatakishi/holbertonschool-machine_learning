#!/usr/bin/env python3
"""
Forward propagation over a convolutional layer of a neural network.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters:
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray of shape (1, 1, 1, c_new)
    - activation: activation function applied to the convolution
    - padding: string that is either 'same' or 'valid'
    - stride: tuple of (sh, sw)

    Returns:
    - The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Padding ölçülərinin təyin edilməsi
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0

    # Giriş matrisinə padding tətbiq edilməsi
    A_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Çıxış matrisinin ölçülərinin hesablanması
    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    # Çıxış matrisinin sıfırlarla başladılması
    Z = np.zeros((m, h_out, w_out, c_new))

    # Konvolusiya əməliyyatının icra edilməsi
    for h in range(h_out):
        for w in range(w_out):
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            # Cari pəncərənin (slice) kəsilməsi
            A_slice = A_pad[:, v_start:v_end, h_start:h_end, :]

            # Element-wise vurma və bütün oxlar üzrə cəmləmə
            Z[:, h, w, :] = np.sum(
                A_slice[:, :, :, :, np.newaxis] * W,
                axis=(1, 2, 3)
            )

    # Bias əlavə edilib aktivasiya funksiyasının tətbiq edilməsi
    return activation(Z + b)
