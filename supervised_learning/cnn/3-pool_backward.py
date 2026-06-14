#!/usr/bin/env python3
"""
Pooling Back Propagation module.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters:
    - dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
      the partial derivatives with respect to the output of the pooling layer
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing
      the output of the previous layer
    - kernel_shape: tuple of (kh, kw) containing the size of the kernel
      for the pooling
    - stride: tuple of (sh, sw) containing the strides for the pooling
    - mode: string containing either 'max' or 'avg'

    Returns:
    - dA_prev: the partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Əvvəlki təbəqənin ölçülərində sıfırlardan ibarət matris yaradılır
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c):
                    # Pəncərənin (slice) koordinatlarının təyini
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    if mode == 'max':
                        # Cari pəncərənin kəsilməsi
                        a_prev_slice = A_prev[i, v_start:v_end,
                                              h_start:h_end, f]
                        # Maksimum elementin yerini təyin edən maska
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        # Qradiyent yalnız maksimum
                        # dəyərin olduğu yerə ötürülür
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            mask * dA[i, h, w, f]
                        )

                    elif mode == 'avg':
                        # Cari qradiyent dəyəri
                        da = dA[i, h, w, f]
                        # Qradiyent pəncərədəki
                        # bütün elementlərə bərabər bölünür
                        average_value = da / (kh * kw)
                        dA_prev[i, v_start:v_end, h_start:h_end, f] += (
                            np.ones((kh, kw)) * average_value
                        )

    return dA_prev
