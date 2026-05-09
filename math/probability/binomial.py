#!/usr/bin/env python3
""" binomial """


class Binomial:
    """ binomial """
    def __init__(self, data=None, n=1, p=0.5):
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("""data must contain multiple values""")
            mean = sum(data) / len(data)
            variance = sum((i - mean) ** 2 for i in data) / len(data)
            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = p
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """ pmf """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        def factorial(n):
            f = 1
            for i in range(1, n + 1):
                f *= i
            return f
        combination = (factorial(self.n) /
                       (factorial(k) * factorial(self.n - k)))
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))
