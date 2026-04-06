#!/usr/bin/env python3
""" sum total """


def summation_i_squared(n):
    """ sum total """
    if isinstance(n, int):
        return sum(i**2 for i in range(1, n+1))
    else:
        return None
