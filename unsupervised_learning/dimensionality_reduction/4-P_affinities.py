#!/usr/bin/env python3
"""
P affinities module for t-SNE.
"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset to be
            transformed by t-SNE
            n is the number of data points
            d is the number of dimensions in each point
        tol (float): the maximum tolerance allowed (inclusive) for the
            difference in Shannon entropy from perplexity for all Gaussian
            distributions
        perplexity (float): the perplexity that all Gaussian distributions
            should have

    Returns:
        numpy.ndarray: P, shape (n, n), containing the symmetric P
            affinities
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low = None
        high = None
        beta = betas[i].copy()

        # i-ci sətirdə özü-özü ilə məsafəni çıxarırıq
        Di = np.append(D[i, :i], D[i, i + 1:])

        Hi, Pi = HP(Di, beta)
        H_diff = Hi - H

        # Entropiya fərqi tolerantlıq daxilinə düşənə qədər
        # beta üzərində binar axtarış aparırıq
        while np.abs(H_diff) > tol:
            if H_diff > 0:
                # Entropiya çox böyükdür -> beta-nı artırmalıyıq
                low = beta[0]
                if high is None:
                    beta[0] = beta[0] * 2
                else:
                    beta[0] = (beta[0] + high) / 2
            else:
                # Entropiya çox kiçikdir -> beta-nı azaltmalıyıq
                high = beta[0]
                if low is None:
                    beta[0] = beta[0] / 2
                else:
                    beta[0] = (beta[0] + low) / 2

            Hi, Pi = HP(Di, beta)
            H_diff = Hi - H

        # Tapılmış Pi dəyərlərini P matrisinin uyğun sətrinə yerləşdiririk
        P[i, :i] = Pi[:i]
        P[i, i + 1:] = Pi[i:]
        betas[i] = beta

    # P affinitilərini simmetrikləşdiririk və normallaşdırırıq
    P = (P + P.T) / (2 * n)

    return P
