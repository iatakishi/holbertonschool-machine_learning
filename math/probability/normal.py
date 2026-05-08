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
