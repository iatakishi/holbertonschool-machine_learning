#!/usr/bin/env python3
""" exponential """


class Exponential:
    """ exponential """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """ pdf """
        if x < 0:
            return 0
        e = 2.7182818285
        P = (self.lambtha * e) ** -(self.lambtha * x)
        return P
