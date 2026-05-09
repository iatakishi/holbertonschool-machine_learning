#!/usr/bin/env python3
""" normal """


class Normal:
    """ normal """
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("""data must contain multiple values""")
            self.mean = sum(data) / len(data)
            deviations = []
            for i in data:
                deviations.append((i - self.mean)**2)
            self.stddev = (sum(deviations) / len(data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ z-score """
        z = (x-self.mean) / self.stddev
        return z

    def x_value(self, z):
        """ x_value """
        x = self.mean + z * self.stddev
        return x

    def pdf(self, x):
        """ pdf """
        pi = 3.1415926536
        e = 2.7182818285
        P = ((1 / (self.stddev * (2 * pi) ** 0.5)) *
             e ** (-(x - self.mean) ** 2 / (2 * self.stddev ** 2)))
        return P

    def cdf(self, x):
        """ cdf """
        pi = 3.1415926536

        def erf(x):
            """ erf """
            erf = (2 / (pi ** 0.5)) * (x - (x ** 3 / 3) +
                                       (x ** 5 / 10) - (x ** 7 / 42) + (x ** 9 / 216))
            return erf
        P = 0.5 * (1 + erf((x - self.mean) / (self.stddev * 2 ** 0.5)))
        return P
