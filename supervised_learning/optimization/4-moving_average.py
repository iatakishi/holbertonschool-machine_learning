#!/usr/bin/env python3
"""Module for calculating weighted moving average"""


# Calculate weighted moving average of a data set
def moving_average(data, beta):
    """Calculate the weighted moving average of a data set with bias correction"""
    v = 0
    averages = []
    for i, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        averages.append(v / (1 - beta ** i))
    return averages
