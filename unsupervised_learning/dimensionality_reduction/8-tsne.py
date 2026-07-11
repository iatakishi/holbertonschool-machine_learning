#!/usr/bin/env python3
"""
t-SNE module.
"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset to be
            transformed by t-SNE
            n is the number of data points
            d is the number of dimensions in each point
        ndims (int): the new dimensional representation of X
        idims (int): the intermediate dimensional representation of X
            after PCA
        perplexity (float): the perplexity
        iterations (int): the number of iterations
        lr (float): the learning rate

    Returns:
        numpy.ndarray: Y, shape (n, ndims), the optimized low dimensional
            transformation of X
    """
    n, d = X.shape

    # Əvvəlcə PCA ilə ölçünü aralıq dimensiyaya endiririk
    X = pca(X, idims)

    # P affinitilərini hesablayırıq
    P = P_affinities(X, perplexity=perplexity)

    # Erkən şişirtmə (early exaggeration)
    P = P * 4

    # Y-i kiçik təsadüfi dəyərlərlə initialize edirik
    Y = np.random.randn(n, ndims)
    Y_prev = Y.copy()

    for i in range(iterations):
        dY, Q = grads(Y, P)

        # Momentum əmsalı: ilk 20 iterasiya üçün 0.5, sonra 0.8
        if i < 20:
            alpha = 0.5
        else:
            alpha = 0.8

        # Qradiyent enişi addımı (məqalədəki səhvə görə burada
        # dY ƏLAVƏ olunur, çıxılmır)
        Y_new = Y + lr * dY + alpha * (Y - Y_prev)

        Y_prev = Y
        Y = Y_new

        # Hər iterasiyadan sonra Y-i ortalamasını çıxaraq mərkəzləşdiririk
        Y = Y - np.mean(Y, axis=0)

        # 100-cü iterasiyada erkən şişirtməni ləğv edirik
        if i == 99:
            P = P / 4

        # Hər 100 iterasiyada (0 daxil olmadan) dəyəri çap edirik
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print('Cost at iteration {}: {}'.format(i + 1, C))

    return Y
